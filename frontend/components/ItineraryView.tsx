"use client";

import dynamic from "next/dynamic";
import type { FinalItinerary } from "@/lib/api";

// Map is client-only (touches window/Leaflet), so load it without SSR.
const MapView = dynamic(() => import("./MapView"), {
  ssr: false,
  loading: () => <div className="map map-loading">Loading map…</div>,
});

const SLOT_TIME: Record<string, string> = {
  d1_breakfast: "Breakfast",
  d1_am_attraction: "Morning",
  d1_lunch: "Lunch",
  d1_pm_attraction: "Afternoon",
  d1_dinner: "Dinner",
  d2_breakfast: "Breakfast",
  d2_am_attraction: "Morning",
  d2_lunch: "Lunch",
  d2_pm_attraction: "Afternoon",
  d2_dinner: "Dinner",
};

function BudgetBar({ budget }: { budget: FinalItinerary["budget"] }) {
  const pct = Math.min(100, Math.round(budget.utilisation * 100));
  return (
    <div className="budget">
      <div className="budget-head">
        <span>
          <strong>${budget.spent.toFixed(0)}</strong> of ${budget.budget_total.toFixed(0)}
        </span>
        <span className="muted">
          {pct}% · {budget.party_size} {budget.party_size === 1 ? "person" : "people"}
        </span>
      </div>
      <div className="budget-track">
        <div
          className={`budget-fill ${budget.over_budget ? "over" : ""}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function ItineraryView({ itinerary }: { itinerary: FinalItinerary }) {
  return (
    <div className="itinerary">
      <div className="itin-head">
        <div>
          <h2>{itinerary.title}</h2>
          <p className="sub">{itinerary.intro}</p>
        </div>
        <span className={`pill ${itinerary.tier === 2 ? "purple" : "wait"}`}>
          Tier {itinerary.tier}
        </span>
      </div>

      <BudgetBar budget={itinerary.budget} />

      <MapView days={itinerary.days} />

      {itinerary.days.map((day) => (
        <div key={day.day} className="day">
          <div className="day-head">
            <h3>Day {day.day}</h3>
            <span className="muted">
              {day.route.total_km} km · {day.route.total_travel_minutes} min travelling
            </span>
          </div>
          {day.summary && <p className="day-summary">{day.summary}</p>}
          <ol className="stops">
            {day.stops.map((stop, i) => (
              <li key={stop.slot} className="stop">
                <span className="stop-num">{i + 1}</span>
                <div className="stop-body">
                  <div className="stop-title">
                    <span className="stop-name">{stop.name}</span>
                    <span className="stop-cost">
                      {stop.cost_per_person ? `$${stop.cost_per_person.toFixed(0)}pp` : "free"}
                    </span>
                  </div>
                  <div className="stop-meta">
                    {SLOT_TIME[stop.slot] ?? stop.slot} · {stop.category.replace(/_/g, " ")} ·{" "}
                    {stop.neighbourhood}
                  </div>
                  {stop.reason && <div className="stop-reason">{stop.reason}</div>}
                </div>
              </li>
            ))}
          </ol>
        </div>
      ))}

      {itinerary.closing && <p className="closing">{itinerary.closing}</p>}

      <div className="itin-foot muted">
        {itinerary.revisions ? `${itinerary.revisions} Critic revision(s) · ` : ""}
        {itinerary.validation.ok ? "all constraints satisfied" : "check the trace for issues"} ·{" "}
        served by {itinerary.mock ? "offline mock" : "live free LLMs"} · {itinerary.elapsed_ms} ms
      </div>
    </div>
  );
}
