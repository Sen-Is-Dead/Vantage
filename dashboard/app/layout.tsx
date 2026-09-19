import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "./nav";

export const metadata: Metadata = {
  title: "Vantage · FPL assistant",
  description: "Read-only view of Vantage's weekly FPL recommendations. Nothing here submits to FPL.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="top">
          <div className="inner">
            <span className="brand">Vantage</span>
            <Nav />
          </div>
        </header>
        <main>{children}</main>
        <footer>Recommendations only. Nothing on this site can submit transfers, lineups or chips to FPL; you apply what you agree with.</footer>
      </body>
    </html>
  );
}
