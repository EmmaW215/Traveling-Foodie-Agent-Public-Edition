"use client";

import { useEffect, useRef, useState } from "react";
import {
  getDatasetMeta,
  getReadiness,
  streamItinerary,
  type DatasetMeta,
  type FinalItinerary,
  type Preferences,
  type Readiness,
  type TraceEvent,
} from "@/lib/api";
import PreferenceForm from "@/components/PreferenceForm";
import TraceTimeline from "@/components/TraceTimeline";
import ItineraryView from "@/components/ItineraryView";
import CopilotChat from "@/components/CopilotChat";

type Mode = "copilot" | "plan1" | "plan2";
type Conn = "waking" | "ok" | "error";

const TABS: { id: Mode; label: string; note: string }[] = [
  { id: "copilot", label: "Ask the guide", note: "Tier 0 · RAG copilot" },
  { id: "plan2", label: "Plan a trip", note: "Tier 2 · multi-agent" },
  { id: "plan1", label: "Plan (simple)", note: "Tier 1 · sequential" },
];

export default function Home() {
  const [conn, setConn] = useState<Conn>("waking");
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [meta, setMeta] = useState<DatasetMeta | null>(null);
  const [connError, setConnError] = useState("");

  const [mode, setMode] = useState<Mode>("copilot");

  // Planner state
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [itinerary, setItinerary] = useState<FinalItinerary | null>(null);
  const [running, setRunning] = useState(false);
  const [planError, setPlanError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setConn("waking");
      try {
        const [r, m] = await Promise.all([getReadiness(), getDatasetMeta()]);
        if (cancelled) return;
        setReadiness(r);
        setMeta(m);
        setConn("ok");
      } catch (e) {
        if (cancelled) return;
        setConnError(e instanceof Error ? e.message : String(e));
        setConn("error");
      }
    })();
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, []);

  const runPlan = async (prefs: Preferences, tier: number) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setEvents([]);
    setItinerary(null);
    setPlanError("");
    setRunning(true);
    try {
      await streamItinerary(
        prefs,
        tier,
        (event) => {
          setEvents((prev) => [...prev, event]);
          if (event.event === "final") setItinerary(event as unknown as FinalItinerary);
          if (event.event === "error") setPlanError(String(event.detail ?? "Pipeline error"));
        },
        ctrl.signal,
      );
    } catch (e) {
      if (!ctrl.signal.aborted) {
        setPlanError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setRunning(false);
    }
  };

  const switchMode = (m: Mode) => {
    if (running) abortRef.current?.abort();
    setMode(m);
  };

  return (
    <main>
      <header className="hdr">
        <div>
          <p className="eyebrow">CALGARY · TRAVELING FOODIE AGENT</p>
          <h1>Plan a food trip, or just ask.</h1>
        </div>
        <div className="status">
          {conn === "waking" && <span className="pill wait">waking backend (~60 s)…</span>}
          {conn === "ok" && readiness && (
            <span className="pill ok">
              online · {readiness.rag_retriever === "upstash" ? "Upstash" : "local"} retriever
            </span>
          )}
          {conn === "error" && <span className="pill warn">backend unreachable</span>}
        </div>
      </header>

      {conn === "error" && (
        <div className="banner error">
          {connError} — if this is the first visit in a while, the free Render instance is cold.
          Reload in a minute.
        </div>
      )}

      {meta && (
        <div className="banner note">
          Demo data: the {meta.restaurants} restaurants and {meta.attractions} attractions are{" "}
          <strong>fictional</strong>, placed on real Calgary geography. {meta.data_disclaimer}
        </div>
      )}

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={mode === t.id}
            className={`tab ${mode === t.id ? "on" : ""}`}
            onClick={() => switchMode(t.id)}
          >
            <span className="tab-label">{t.label}</span>
            <span className="tab-note">{t.note}</span>
          </button>
        ))}
      </nav>

      {mode === "copilot" ? (
        <section className="panel">
          <CopilotChat />
        </section>
      ) : (
        <section className="panel">
          <PreferenceForm
            cuisineOptions={meta?.cuisines ?? DEFAULT_CUISINES}
            disabled={running || conn !== "ok"}
            onSubmit={(prefs) => runPlan(prefs, mode === "plan2" ? 2 : 1)}
          />
          {planError && <div className="banner error">{planError}</div>}
          <TraceTimeline events={events} running={running} />
          {itinerary && <ItineraryView itinerary={itinerary} />}
        </section>
      )}

      <footer>
        Zero-cost stack — Vercel · Render · Upstash Vector · Groq / Gemini / OpenRouter free tiers ·
        OpenStreetMap. All three hackathon tiers, live.
      </footer>
    </main>
  );
}

const DEFAULT_CUISINES = [
  "japanese",
  "italian",
  "thai",
  "chinese",
  "indian",
  "mexican",
  "french",
  "korean",
];
