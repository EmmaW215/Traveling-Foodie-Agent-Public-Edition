"use client";

import { useEffect, useRef } from "react";
import type { ItineraryDay } from "@/lib/api";

/**
 * Leaflet over OpenStreetMap tiles — free, no API key, the $0 replacement for
 * Google Maps. Leaflet is loaded from the CDN inside this client-only component
 * so there are no bundler, SSR or marker-asset headaches: we touch `window.L`
 * only after the script has loaded, in an effect that never runs on the server.
 *
 * Numbered divIcon markers (colour-coded per day) sidestep the classic Leaflet
 * default-marker image problem entirely.
 */

const CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
const JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
const DAY_COLOURS = ["#4b286d", "#2b8000", "#c75300", "#065a82"];

// Minimal shape of the Leaflet globals we use — avoids pulling in @types/leaflet.
type LeafletMap = { remove: () => void; fitBounds: (b: unknown, o?: unknown) => void };
type Leaflet = {
  map: (el: HTMLElement, opts?: unknown) => LeafletMap & Record<string, (...a: unknown[]) => unknown>;
  tileLayer: (url: string, opts?: unknown) => { addTo: (m: unknown) => unknown };
  marker: (ll: [number, number], opts?: unknown) => { addTo: (m: unknown) => { bindPopup: (s: string) => unknown } };
  polyline: (lls: [number, number][], opts?: unknown) => { addTo: (m: unknown) => unknown };
  divIcon: (opts: unknown) => unknown;
  latLngBounds: (lls: [number, number][]) => unknown;
};

declare global {
  interface Window {
    L?: Leaflet;
  }
}

let loader: Promise<Leaflet> | null = null;

function loadLeaflet(): Promise<Leaflet> {
  if (typeof window === "undefined") return Promise.reject(new Error("no window"));
  if (window.L) return Promise.resolve(window.L);
  if (loader) return loader;
  loader = new Promise<Leaflet>((resolve, reject) => {
    if (!document.querySelector(`link[href="${CSS}"]`)) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = CSS;
      document.head.appendChild(link);
    }
    const script = document.createElement("script");
    script.src = JS;
    script.async = true;
    script.onload = () => (window.L ? resolve(window.L) : reject(new Error("Leaflet failed to load")));
    script.onerror = () => reject(new Error("Leaflet failed to load"));
    document.head.appendChild(script);
  });
  return loader;
}

export default function MapView({ days }: { days: ItineraryDay[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LeafletMap | null>(null);

  useEffect(() => {
    let cancelled = false;

    loadLeaflet()
      .then((L) => {
        if (cancelled || !ref.current) return;
        // Tear down a previous instance (re-planning re-renders this).
        if (mapRef.current) {
          mapRef.current.remove();
          mapRef.current = null;
        }

        const map = L.map(ref.current, { scrollWheelZoom: false });
        mapRef.current = map;
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "&copy; OpenStreetMap contributors",
          maxZoom: 19,
        }).addTo(map);

        const allPoints: [number, number][] = [];

        days.forEach((day, di) => {
          const colour = DAY_COLOURS[di % DAY_COLOURS.length];
          const line: [number, number][] = [];
          day.stops.forEach((stop, si) => {
            const ll: [number, number] = [stop.lat, stop.lon];
            allPoints.push(ll);
            line.push(ll);
            const icon = L.divIcon({
              className: "map-pin",
              html: `<span style="background:${colour}">${si + 1}</span>`,
              iconSize: [26, 26],
              iconAnchor: [13, 13],
            });
            L.marker(ll, { icon })
              .addTo(map)
              .bindPopup(
                `<strong>${stop.name}</strong><br/>${stop.category} · ${
                  stop.cost_per_person ? "$" + stop.cost_per_person + "pp" : "free"
                }<br/><em>Day ${day.day}, stop ${si + 1}</em>`,
              );
          });
          if (line.length > 1) {
            L.polyline(line, { color: colour, weight: 3, opacity: 0.6, dashArray: "6 6" }).addTo(map);
          }
        });

        if (allPoints.length) {
          map.fitBounds(L.latLngBounds(allPoints), { padding: [30, 30] });
        }
      })
      .catch(() => {
        /* map is a nice-to-have; the itinerary still renders without it */
      });

    return () => {
      cancelled = true;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [days]);

  return <div ref={ref} className="map" role="img" aria-label="Map of the itinerary stops" />;
}
