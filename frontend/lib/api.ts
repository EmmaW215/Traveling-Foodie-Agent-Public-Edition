/**
 * Client for the Render backend.
 *
 * The base URL is injected at build time via NEXT_PUBLIC_API_BASE (a Vercel env
 * var). Everything runs from the browser straight to Render — the frontend does
 * no server-side work, which is what keeps Vercel function usage near zero.
 */

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");

// ---------------------------------------------------------------------------
// Types (mirror the backend contracts)
// ---------------------------------------------------------------------------
export type Health = {
  status: string;
  app: string;
  version: string;
  uptime_s: number;
};

export type Readiness = {
  llm_providers: string[];
  llm_configured: boolean;
  embeddings_configured: boolean;
  vector_db_configured: boolean;
  rag_retriever: string;
  dataset_ready: boolean;
  default_tier: number;
  allowed_origins: string[];
};

export type DatasetMeta = {
  cities: string[];
  restaurants: number;
  attractions: number;
  cuisines: string[];
  neighbourhoods: string[];
  data_version: string;
  data_disclaimer: string;
};

export type Preferences = {
  city: string;
  days: number;
  budget_total: number;
  party_size: number;
  cuisines: string[];
  allergies: string[];
  notes: string;
};

export type Stop = {
  slot: string;
  venue_id: string;
  name: string;
  category: string;
  kind: string;
  neighbourhood: string;
  cost_per_person: number;
  lat: number;
  lon: number;
  reason: string;
};

export type Leg = {
  from_name: string;
  to_name: string;
  distance_km: number;
  mode: string;
  minutes: number;
};

export type Route = {
  stops: { venue_id: string; name: string; lat: number; lon: number; kind: string }[];
  legs: Leg[];
  total_km: number;
  total_travel_minutes: number;
};

export type Budget = {
  budget_total: number;
  spent: number;
  remaining: number;
  party_size: number;
  over_budget: boolean;
  utilisation: number;
  line_items: { slot: string; name: string; cost_per_person: number; total: number }[];
};

export type ItineraryDay = {
  day: number;
  summary: string;
  stops: Stop[];
  route: Route;
};

export type FinalItinerary = {
  event: "final";
  tier: number;
  title: string;
  intro: string;
  closing: string;
  days: ItineraryDay[];
  budget: Budget;
  validation: { ok: boolean; issues: { slot: string; issue: string; detail: string }[] };
  data_version: string;
  elapsed_ms: number;
  mock: boolean;
  revisions?: number;
};

// A trace frame is any of the streamed events; `event` names the kind.
export type TraceEvent = {
  event: string;
  [key: string]: unknown;
};

export type Citation = {
  venue_id: string;
  name: string;
  category: string;
  neighbourhood: string;
  kind: string;
};

export type CopilotAnswer = {
  answer: string;
  grounded: boolean;
  refused: boolean;
  citations: Citation[];
  sources: string[];
  retriever: string;
  mock: boolean;
};

// ---------------------------------------------------------------------------
// Plain JSON calls with cold-start tolerance
// ---------------------------------------------------------------------------
export class ApiError extends Error {}

/**
 * Render free instances sleep after 15 min idle and take ~60 s to wake, so we
 * retry with a long ceiling. The caller can show a "waking up" state instead of
 * a hard failure on the first visit of the day.
 */
async function fetchWithWake<T>(path: string, init?: RequestInit, attempts = 3): Promise<T> {
  if (!API_BASE) {
    throw new ApiError("NEXT_PUBLIC_API_BASE is not set. Configure it in Vercel project settings.");
  }
  let lastError: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 80_000);
      const res = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
      clearTimeout(timer);
      if (!res.ok) {
        const detail = await res.text().catch(() => res.statusText);
        throw new ApiError(`HTTP ${res.status}: ${detail.slice(0, 200)}`);
      }
      return (await res.json()) as T;
    } catch (err) {
      lastError = err;
      if (i < attempts - 1) await new Promise((r) => setTimeout(r, 3000 * (i + 1)));
    }
  }
  throw lastError instanceof Error ? lastError : new ApiError("Request failed");
}

export const getHealth = () => fetchWithWake<Health>("/health");
export const getReadiness = () => fetchWithWake<Readiness>("/readiness");
export const getDatasetMeta = () => fetchWithWake<DatasetMeta>("/dataset/meta");

export const askCopilot = (question: string) =>
  fetchWithWake<CopilotAnswer>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

// ---------------------------------------------------------------------------
// Server-Sent Events: /itinerary streams the agent trace, then a final frame
// ---------------------------------------------------------------------------
/**
 * Stream the itinerary pipeline. `/itinerary` is a POST that returns
 * text/event-stream, so the browser's EventSource (GET-only) can't be used —
 * we read the body as a stream and parse `data:` frames ourselves.
 *
 * Calls `onEvent` for every trace frame as it arrives. Resolves when the stream
 * ends. The AbortSignal lets the caller cancel an in-flight plan.
 */
export async function streamItinerary(
  prefs: Preferences,
  tier: number,
  onEvent: (event: TraceEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (!API_BASE) {
    throw new ApiError("NEXT_PUBLIC_API_BASE is not set. Configure it in Vercel project settings.");
  }

  const res = await fetch(`${API_BASE}/itinerary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ preferences: prefs, tier }),
    signal,
  });

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => res.statusText);
    throw new ApiError(`HTTP ${res.status}: ${detail.slice(0, 200)}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // SSE frames are separated by a blank line; each frame's payload lives on
  // one or more `data:` lines.
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const dataLines = frame
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim());
      if (!dataLines.length) continue;
      try {
        onEvent(JSON.parse(dataLines.join("\n")) as TraceEvent);
      } catch {
        // Ignore a malformed frame rather than killing the whole stream.
      }
    }
  }
}
