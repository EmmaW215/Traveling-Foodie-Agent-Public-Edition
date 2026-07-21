# Dataset — provenance and honesty notes

## What this data is

**The venues in this dataset are fictional.** The geography is real: every venue sits in a real
Calgary neighbourhood at plausible real coordinates, so distances, walking times and route ordering
all behave exactly as they would with a live catalogue.

| Field group | Real or synthetic |
|---|---|
| Neighbourhoods, coordinates, distances | **Real** — actual Calgary neighbourhood centroids |
| Venue names | Synthetic |
| Cuisine, price band, cost per person | Synthetic but internally consistent |
| Opening hours, closed days | Synthetic |
| Allergen flags, dietary options | Synthetic |
| Ratings | Synthetic |

## Why not real restaurants?

This is a public, deployed application. Publishing invented opening hours, prices or — most
seriously — **allergen information** attached to real, named businesses would misinform anyone who
trusted it, and could put someone with a peanut allergy at risk. Restaurant hours and menus also
change constantly, so a hardcoded snapshot of real venues would be wrong within weeks and there is
no free API budget to refresh it.

A synthetic catalogue over real geography gives us everything the agent architecture needs to be
genuinely exercised — distance-aware routing, budget arithmetic, opening-hours conflicts, allergen
exclusion — with zero risk of misleading a real user about a real business.

The dataset is labelled as demo data in three places: this file, the `data_disclaimer` field
returned by `GET /dataset/meta`, and the UI footer.

## Swapping in real data later

The schema is deliberately compatible with OpenStreetMap tags, so a real-data import is a drop-in
replacement rather than a rewrite:

| Our column | OSM tag |
|---|---|
| `name` | `name` |
| `lat` / `lon` | node position |
| `cuisine` | `cuisine` |
| `open_time` / `close_time` / `closed_days` | `opening_hours` (needs parsing) |
| `dietary_options` | `diet:vegetarian`, `diet:vegan`, `diet:halal`, `diet:gluten_free` |

Example Overpass query for downtown Calgary restaurants:

```
[out:csv(name,::lat,::lon,cuisine,opening_hours)];
node[amenity=restaurant][name](51.036,-114.095,51.056,-114.048);
out 200;
```

OSM data is ODbL-licensed: attribution required, and derived databases must stay open. Note that
OSM has **no price or allergen data**, so those columns would still need a documented estimation
method — which is the harder half of the problem, and the reason v1 does not attempt it.

## Planted edge cases

Three venues exist to make the guards testable — the same technique as the hackathon starter kit.
They are flagged in the `is_trap` column and asserted in `tests/test_guards.py`:

| `is_trap` | Venue | What it proves |
|---|---|---|
| `peanut_risk` | r008 Peanut Garden Thai | Must be excluded whenever `peanut` is in the allergy list, even though it is otherwise a great match |
| `budget_buster` | r003 Ember & Oak Steakhouse, r049 Mount Royal Fine Dining | A single pick blows a $500 two-day budget; the budget guard must catch it |
| `closed_monday` | r005 Jade Lantern Dim Sum, a002 Prairie Heritage Museum | Must not be scheduled on a Monday slot |

Removing or renaming these rows will fail the test suite by design.

## Rebuilding

```bash
cd backend
python -m scripts.seed          # CSVs -> data/build/{foodie.sqlite,distance_matrix.json,chunks.jsonl}
```

Build artefacts are gitignored and regenerated on every deploy — the CSVs are the only source of
truth.

## Licence

The CSVs in this directory are released under the same MIT licence as the repository. They contain
no third-party data.
