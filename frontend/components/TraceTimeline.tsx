"use client";

import type { TraceEvent } from "@/lib/api";

/**
 * The agent-trace timeline — the public equivalent of the hackathon's
 * "UI shows the agent trace" requirement. Renders each streamed frame as a row
 * so the user watches the planner, the parallel executors, the Critic loop and
 * the revisions happen live.
 */

const SLOT_LABELS: Record<string, string> = {
  d1_breakfast: "Day 1 · breakfast",
  d1_am_attraction: "Day 1 · morning",
  d1_lunch: "Day 1 · lunch",
  d1_pm_attraction: "Day 1 · afternoon",
  d1_dinner: "Day 1 · dinner",
  d2_breakfast: "Day 2 · breakfast",
  d2_am_attraction: "Day 2 · morning",
  d2_lunch: "Day 2 · lunch",
  d2_pm_attraction: "Day 2 · afternoon",
  d2_dinner: "Day 2 · dinner",
};

function slotLabel(slot: unknown): string {
  return typeof slot === "string" ? SLOT_LABELS[slot] ?? slot : "";
}

type Row = { icon: string; tone: string; title: string; detail?: string };

function toRow(e: TraceEvent): Row | null {
  switch (e.event) {
    case "planner_start":
      return { icon: "◍", tone: "purple", title: "Planner", detail: String(e.message ?? "") };
    case "planner_done": {
      const c = (e.cuisines_priority as string[]) ?? [];
      return {
        icon: "◍",
        tone: "purple",
        title: "Planner ready",
        detail: c.length ? `Prioritising ${c.join(", ")}` : String(e.summary ?? ""),
      };
    }
    case "executors_dispatched":
      return {
        icon: "⇉",
        tone: "purple",
        title: `Dispatched ${(e.slots as string[])?.length ?? 0} slots in parallel`,
      };
    case "executor_start":
      return { icon: "·", tone: "muted", title: `Choosing ${slotLabel(e.slot)}…` };
    case "executor_result":
      return {
        icon: "✓",
        tone: "green",
        title: slotLabel(e.slot),
        detail: `${e.name}${e.picked_by === "fallback" ? " (fallback)" : ""}`,
      };
    case "critic_reviewed": {
      const issues = (e.issues as { issue: string }[]) ?? [];
      const tags = Array.from(new Set(issues.map((i) => i.issue))).join(", ");
      return {
        icon: "⟳",
        tone: issues.length ? "orange" : "green",
        title: `Critic · pass ${e.iteration}`,
        detail: issues.length ? `${issues.length} issue(s): ${tags}` : "clean — plan looks good",
      };
    }
    case "revision":
      return {
        icon: "✎",
        tone: "orange",
        title: `Revised ${slotLabel(e.slot)}`,
        detail: `${e.replaced} → ${e.with_} (${e.issue})`,
      };
    case "validation":
      return {
        icon: e.ok ? "✓" : "!",
        tone: e.ok ? "green" : "orange",
        title: e.ok ? "Validated — all constraints satisfied" : "Validation found issues",
        detail: e.ok ? undefined : JSON.stringify(e.issues),
      };
    case "route_ready":
      return { icon: "⇢", tone: "purple", title: `Route planned · ${e.total_km} km total` };
    case "notice":
      return { icon: "i", tone: "muted", title: String(e.message ?? "") };
    case "error":
      return { icon: "✕", tone: "red", title: "Error", detail: String(e.detail ?? "") };
    default:
      return null;
  }
}

export default function TraceTimeline({
  events,
  running,
}: {
  events: TraceEvent[];
  running: boolean;
}) {
  const rows = events.map(toRow).filter((r): r is Row => r !== null);
  if (!rows.length && !running) return null;

  return (
    <div className="trace">
      <div className="trace-head">
        <span className="trace-title">Agent trace</span>
        {running && <span className="pill wait">working…</span>}
      </div>
      <ol className="trace-list">
        {rows.map((r, i) => (
          <li key={i} className={`trace-row tone-${r.tone}`}>
            <span className="trace-icon" aria-hidden>
              {r.icon}
            </span>
            <span className="trace-body">
              <span className="trace-row-title">{r.title}</span>
              {r.detail && <span className="trace-row-detail">{r.detail}</span>}
            </span>
          </li>
        ))}
        {running && (
          <li className="trace-row tone-muted">
            <span className="trace-icon spin" aria-hidden>
              ◌
            </span>
            <span className="trace-body">
              <span className="trace-row-title">thinking…</span>
            </span>
          </li>
        )}
      </ol>
    </div>
  );
}
