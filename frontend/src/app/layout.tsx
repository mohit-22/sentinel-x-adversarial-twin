import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const NAV_LINKS: { href: string; label: string }[] = [
  { href: "/", label: "Command Center" },
  { href: "/red-team", label: "Red Team Lab" },
  { href: "/payment-twin", label: "Payment Twin" },
  { href: "/blue-team-soc", label: "Blue Team SOC" },
  { href: "/arena", label: "Adversarial Arena" },
  { href: "/judge", label: "Judge Sandbox" },
  { href: "/observatory", label: "Threat Observatory" },
];

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Sentinel-X",
  description: "Autonomous Adversarial Payment Twin",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <nav className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-border bg-background px-4 py-2 text-xs text-muted-foreground">
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {NAV_LINKS.map((link) => (
              <Link key={link.href} href={link.href} className="hover:text-foreground">
                {link.label}
              </Link>
            ))}
          </div>
          <span
            className="rounded border px-1.5 py-0.5 text-[10px] font-medium"
            style={{ borderColor: "var(--neon-green)", color: "var(--neon-green)" }}
          >
            AI Investigator: Active
          </span>
        </nav>
        {children}
      </body>
    </html>
  );
}
