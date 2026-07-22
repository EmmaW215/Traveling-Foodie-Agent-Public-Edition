"use client";

import { useRef, useState } from "react";
import { askCopilot, type CopilotAnswer } from "@/lib/api";

/**
 * Tier 0 — the RAG copilot. Ask a question about the Calgary guide; get an
 * answer grounded in it, with citations, or a refusal. This is a plain Q&A
 * chat, distinct from the itinerary planner.
 */

type Turn = { question: string; answer?: CopilotAnswer; error?: string };

const SUGGESTIONS = [
  "Where can I get good ramen for lunch?",
  "I have a peanut allergy — safe dinner spots?",
  "What free attractions are near downtown?",
];

export default function CopilotChat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const ask = async (question: string) => {
    const text = question.trim();
    if (!text || busy) return;
    setBusy(true);
    setQ("");
    setTurns((t) => [...t, { question: text }]);
    try {
      const answer = await askCopilot(text);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, answer } : turn)));
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Something went wrong.";
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error: msg } : turn)));
    } finally {
      setBusy(false);
      requestAnimationFrame(() => listRef.current?.scrollTo(0, listRef.current.scrollHeight));
    }
  };

  return (
    <div className="copilot">
      <p className="sub">
        Ask about restaurants, cuisines, neighbourhoods, budgets, dietary needs, or attractions. The
        copilot answers <strong>only</strong> from the Calgary guide and cites what it used.
      </p>

      {!turns.length && (
        <div className="suggestions">
          {SUGGESTIONS.map((s) => (
            <button key={s} className="chip" onClick={() => ask(s)} disabled={busy}>
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="chat-list" ref={listRef}>
        {turns.map((t, i) => (
          <div key={i} className="chat-turn">
            <div className="bubble user">{t.question}</div>
            {t.answer && (
              <div className={`bubble bot ${t.answer.refused ? "refused" : ""}`}>
                <p>{t.answer.answer}</p>
                {t.answer.citations.length > 0 && (
                  <div className="cites">
                    {t.answer.citations.map((c) => (
                      <span key={c.venue_id} className="cite">
                        {c.name} · {c.category.replace(/_/g, " ")}
                      </span>
                    ))}
                  </div>
                )}
                <div className="bubble-foot">
                  {t.answer.refused ? "no grounded answer" : "grounded"} · via {t.answer.retriever}
                  {t.answer.mock ? " · mock" : ""}
                </div>
              </div>
            )}
            {t.error && <div className="bubble bot refused">{t.error}</div>}
            {!t.answer && !t.error && <div className="bubble bot">thinking…</div>}
          </div>
        ))}
      </div>

      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          ask(q);
        }}
      >
        <input
          type="text"
          value={q}
          maxLength={500}
          placeholder="Ask the copilot…"
          onChange={(e) => setQ(e.target.value)}
          disabled={busy}
        />
        <button type="submit" disabled={busy || q.trim().length < 2}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
    </div>
  );
}
