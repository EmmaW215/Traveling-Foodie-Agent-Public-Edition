"use client";

import { useEffect, useState } from "react";
import { API_BASE, getHealth, getReadiness, postEcho } from "@/lib/api";
import type { EchoResult, Health, Readiness } from "@/lib/api";

type Phase = "idle" | "waking" | "ok" | "error";

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [health, setHealth] = useState<Health | null>(null);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [error, setError] = useState<string>("");

  const [message, setMessage] = useState("ping");
  const [echo, setEcho] = useState<EchoResult | null>(null);
  const [echoBusy, setEchoBusy] = useState(false);
  const [echoError, setEchoError] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setPhase("waking");
      try {
        const [h, r] = await Promise.all([getHealth(), getReadiness()]);
        if (cancelled) return;
        setHealth(h);
        setReadiness(r);
        setPhase("ok");
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setPhase("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function runEcho() {
    setEchoBusy(true);
    setEchoError("");
    setEcho(null);
    try {
      setEcho(await postEcho(message));
    } catch (e) {
      setEchoError(e instanceof Error ? e.message : String(e));
    } finally {
      setEchoBusy(false);
    }
  }

  return (
    <main>
      <p className="eyebrow">M0 · SCAFFOLD</p>
      <h1>Traveling Foodie Agent</h1>
      <p className="sub">
        Public edition — an agentic AI travel concierge. This page is the M0 proof of life: it
        confirms the Vercel frontend can reach the Render backend and that the free LLM fallback
        chain answers.
      </p>

      <section className="card">
        <h2>Backend connection</h2>
        <div className="row">
          <span className="label">API base</span>
          <span className="value">{API_BASE || "(not configured)"}</span>
        </div>
        <div className="row">
          <span className="label">Status</span>
          <span className="value">
            {phase === "waking" && <span className="pill wait">waking up (~60 s)…</span>}
            {phase === "ok" && <span className="pill ok">connected</span>}
            {phase === "error" && <span className="pill warn">unreachable</span>}
            {phase === "idle" && <span className="pill wait">starting…</span>}
          </span>
        </div>
        {health && (
          <>
            <div className="row">
              <span className="label">Version</span>
              <span className="value">{health.version}</span>
            </div>
            <div className="row">
              <span className="label">Uptime</span>
              <span className="value">{health.uptime_s}s</span>
            </div>
          </>
        )}
        {phase === "error" && (
          <p className="note">
            {error} — if this is the first request in a while, the free Render instance is cold;
            reload in a minute.
          </p>
        )}
      </section>

      {readiness && (
        <section className="card">
          <h2>Readiness</h2>
          <div className="row">
            <span className="label">LLM providers</span>
            <span className="value">
              {readiness.llm_providers.length ? readiness.llm_providers.join(" → ") : "none"}
            </span>
          </div>
          <div className="row">
            <span className="label">Embeddings (Gemini)</span>
            <span className="value">
              <span className={readiness.embeddings_configured ? "pill ok" : "pill warn"}>
                {readiness.embeddings_configured ? "configured" : "missing"}
              </span>
            </span>
          </div>
          <div className="row">
            <span className="label">Vector DB (Upstash)</span>
            <span className="value">
              <span className={readiness.vector_db_configured ? "pill ok" : "pill warn"}>
                {readiness.vector_db_configured ? "configured" : "missing"}
              </span>
            </span>
          </div>
          <div className="row">
            <span className="label">Default tier</span>
            <span className="value">Tier {readiness.default_tier}</span>
          </div>
        </section>
      )}

      <section className="card">
        <h2>LLM round-trip test</h2>
        <div className="stack">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Say something to the agent"
          />
          <button onClick={runEcho} disabled={echoBusy || phase !== "ok"}>
            {echoBusy ? "Calling…" : "Send"}
          </button>
        </div>
        {echo && (
          <pre>
            {echo.reply}
            {"\n\n"}— served by {echo.served_by} ({echo.model})
          </pre>
        )}
        {echoError && <p className="note">{echoError}</p>}
        <p className="note">
          Tier 0 (RAG copilot), Tier 1 (scripted agent) and Tier 2 (multi-agent) replace this panel
          in M2–M4.
        </p>
      </section>

      <footer>
        Zero-cost stack — Vercel · Render · GitHub Actions · Upstash Vector · Groq / Gemini /
        OpenRouter free tiers.
      </footer>
    </main>
  );
}
