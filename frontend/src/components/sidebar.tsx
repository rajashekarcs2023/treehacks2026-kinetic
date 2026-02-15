"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Video,
  Trophy,
  History,
  Settings,
  Zap,
  Mic,
  Eye,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", icon: LayoutDashboard, label: "Dashboard" },
  { href: "/coach", icon: Video, label: "Coach" },
  { href: "/skills", icon: Trophy, label: "Skills" },
  { href: "/history", icon: History, label: "History" },
  { href: "/monitor", icon: Eye, label: "Monitor" },
  { href: "/settings", icon: Settings, label: "Settings" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-[180px] flex-col border-r border-border bg-sidebar py-4 px-3">
      {/* Logo */}
      <Link
        href="/"
        className="mb-8 flex items-center gap-2.5 px-2"
      >
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground">
          <Zap className="h-4 w-4" />
        </div>
        <span className="text-sm font-bold tracking-tight">Kinetic AI</span>
      </Link>

      {/* Nav Items */}
      <nav className="flex flex-1 flex-col gap-1">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-primary/15 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )}
            >
              <item.icon className="h-4 w-4 shrink-0" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Voice indicator at bottom */}
      <div className="mt-auto px-2">
        <button className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-muted-foreground transition-all hover:bg-accent hover:text-foreground">
          <Mic className="h-4 w-4 shrink-0" />
          Voice
        </button>
      </div>
    </aside>
  );
}
