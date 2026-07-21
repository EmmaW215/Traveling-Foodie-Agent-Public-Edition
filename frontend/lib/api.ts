/**
 * Thin client for the Render backend.
 * The base URL is injected at build time via NEXT_PUBLIC_API_BASE (Vercel env var).
 */

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");

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
  default_tier: number;
  allowed_origins: string[];
};

export type EchoResult = {
  reply: string;
  served_by: string;
  model: string;
};

/**
 * Render free instances sleep after 15 min idle and take ~60 s to wake.
 * We retry with a long ceiling so the UI can show a "waking up" state
 * instead of a hard failure on the first visit of the day.
 */
async function fetchWithWake<T>(path: string, init?: RequestInit, attempts = 3): Promise<T> {
  if (!API_BASE) {
    throw new Error("NEXT_PUBLIC_API_BASE is not set. Configure it in Vercel project settings.");
  }

  let lastError: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 75_000);
      const res = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
      clearTimeout(timer);

      if (!res.ok) {
        const detail = await res.text().catch(() => res.statusText);
        throw new Error(`HTTP ${res.status}: ${detail.slice(0, 200)}`);
      }
      return (await res.json()) as T;
    } catch (err) {
      lastError = err;
      if (i < attempts - 1) {
        await new Promise((r) => setTimeout(r, 3000 * (i + 1)));
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Request failed");
}

export const getHealth = () => fetchWithWake<Health>("/health");

export const getReadiness = () => fetchWithWake<Readiness>("/readiness");

export const postEcho = (message: string) =>
  fetchWithWake<EchoResult>("/echo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
