import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Traveling Foodie Agent",
  description:
    "An agentic AI travel concierge that plans a reasoned 2-day food itinerary — public edition.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
