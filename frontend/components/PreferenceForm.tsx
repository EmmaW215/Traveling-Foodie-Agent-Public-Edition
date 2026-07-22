"use client";

import { useState } from "react";
import type { Preferences } from "@/lib/api";

/**
 * The traveller's request. City is fixed to Calgary (the only seeded city);
 * cuisines and allergies are chips toggled from what the dataset actually
 * carries, so the form can't ask for something the agent can't honour.
 */

const ALLERGENS = ["peanut", "tree_nut", "dairy", "gluten", "egg", "soy", "shellfish", "fish", "sesame"];

export default function PreferenceForm({
  cuisineOptions,
  disabled,
  onSubmit,
}: {
  cuisineOptions: string[];
  disabled: boolean;
  onSubmit: (prefs: Preferences) => void;
}) {
  const [days, setDays] = useState(2);
  const [budget, setBudget] = useState(500);
  const [party, setParty] = useState(2);
  const [cuisines, setCuisines] = useState<string[]>(["japanese", "italian"]);
  const [allergies, setAllergies] = useState<string[]>(["peanut"]);
  const [notes, setNotes] = useState("");

  const toggle = (list: string[], set: (v: string[]) => void, value: string) =>
    set(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      city: "Calgary",
      days,
      budget_total: budget,
      party_size: party,
      cuisines,
      allergies,
      notes: notes.trim(),
    });
  };

  return (
    <form className="form" onSubmit={submit}>
      <div className="form-grid">
        <label className="field">
          <span>Days</span>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} disabled={disabled}>
            <option value={1}>1 day</option>
            <option value={2}>2 days</option>
          </select>
        </label>
        <label className="field">
          <span>Budget (total, CAD)</span>
          <input
            type="number"
            min={50}
            step={10}
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
            disabled={disabled}
          />
        </label>
        <label className="field">
          <span>Party size</span>
          <input
            type="number"
            min={1}
            max={8}
            value={party}
            onChange={(e) => setParty(Number(e.target.value))}
            disabled={disabled}
          />
        </label>
      </div>

      <fieldset className="chips" disabled={disabled}>
        <legend>Cuisines you love</legend>
        <div className="chip-row">
          {cuisineOptions.map((c) => (
            <button
              type="button"
              key={c}
              className={`chip ${cuisines.includes(c) ? "on" : ""}`}
              onClick={() => toggle(cuisines, setCuisines, c)}
            >
              {c.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="chips" disabled={disabled}>
        <legend>Allergies (these venues are excluded)</legend>
        <div className="chip-row">
          {ALLERGENS.map((a) => (
            <button
              type="button"
              key={a}
              className={`chip danger ${allergies.includes(a) ? "on" : ""}`}
              onClick={() => toggle(allergies, setAllergies, a)}
            >
              {a.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </fieldset>

      <label className="field">
        <span>Anything else? (vibe, must-dos)</span>
        <input
          type="text"
          value={notes}
          maxLength={200}
          placeholder="e.g. romantic, walkable, great coffee"
          onChange={(e) => setNotes(e.target.value)}
          disabled={disabled}
        />
      </label>

      <button className="cta" type="submit" disabled={disabled}>
        {disabled ? "Planning…" : "Plan my trip"}
      </button>
    </form>
  );
}
