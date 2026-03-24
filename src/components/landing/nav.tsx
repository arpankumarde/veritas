"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

export default function LandingNav() {
  const router = useRouter();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-obs-border">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4 md:px-10">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-9 h-9 rounded-lg bg-amber-soft border border-amber/10">
            <span className="material-symbols-outlined text-amber text-xl font-bold">verified</span>
          </div>
          <div>
            <h1 className="text-lg font-display font-semibold tracking-tight text-text">Veritas</h1>
            <p className="text-[10px] text-text-secondary font-mono uppercase tracking-[0.2em] leading-none">Hierarchical Fact Checking</p>
          </div>
        </div>

        <div className="hidden md:flex items-center gap-8 text-[13px] font-medium text-text-secondary uppercase tracking-wider">
          <Link href="#process" className="hover:text-amber transition-colors">Process</Link>
          <Link href="#features" className="hover:text-amber transition-colors">Technology</Link>
          <Link href="/dashboard" className="hover:text-amber transition-colors">Dashboard</Link>
        </div>

        <button
          onClick={() => router.push("/dashboard")}
          className="obs-btn btn-primary rounded-lg px-6 py-2 text-[13px] tracking-wide bg-amber hover:bg-amber-hover text-white shadow-none"
        >
          Get Started
        </button>
      </div>
    </nav>
  );
}
