"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  ["/", "Recommendation"],
  ["/predictions", "Predictions"],
  ["/accuracy", "Accuracy"],
  ["/simulate", "Simulate"],
];

export function Nav() {
  const path = usePathname();
  return (
    <nav>
      {links.map(([href, label]) => (
        <Link key={href} href={href} className={path === href ? "active" : ""}>{label}</Link>
      ))}
    </nav>
  );
}
