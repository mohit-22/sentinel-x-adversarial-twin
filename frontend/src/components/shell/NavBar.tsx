"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Swords,
  Users,
  ShieldCheck,
  FlaskConical,
  Gavel,
  Telescope,
} from "lucide-react";

const NAV_LINKS = [
  { href: "/", label: "Command Center", icon: LayoutDashboard },
  { href: "/red-team", label: "Red Team", icon: Swords },
  { href: "/payment-twin", label: "Payment Twin", icon: Users },
  { href: "/blue-team-soc", label: "Blue Team SOC", icon: ShieldCheck },
  { href: "/arena", label: "Arena", icon: FlaskConical },
  { href: "/judge", label: "Judge Mode", icon: Gavel },
  { href: "/observatory", label: "Threat Observatory", icon: Telescope },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="flex flex-wrap gap-1 border-b border-[var(--surface-glass-border)] bg-[var(--surface-glass)] px-3 py-1.5 backdrop-blur-md">
      {NAV_LINKS.map((link) => {
        const active = pathname === link.href;
        const Icon = link.icon;
        return (
          <Link
            key={link.href}
            href={link.href}
            className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
              active
                ? "bg-white/[0.06] text-foreground"
                : "text-muted-foreground hover:bg-white/[0.03] hover:text-foreground"
            }`}
            style={active ? { boxShadow: "inset 0 0 0 1px var(--surface-glass-border)" } : undefined}
          >
            <Icon
              className="h-3.5 w-3.5"
              style={{ color: active ? "var(--neon-green)" : undefined }}
              aria-hidden="true"
            />
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
