"use client";

import { useRouter } from "next/navigation";

export default function LandingHero() {
  const router = useRouter();

  const logos = [
    { name: "Semantic Scholar", icon: "school" },
    { name: "arXiv.org", icon: "menu_book" },
    { name: "Bright Data", icon: "dataset" },
    { name: "PubMed", icon: "health_and_safety" },
    { name: "IEEE Xplore", icon: "terminal" },
    { name: "JSTOR", icon: "library_books" },
  ];

  return (
    <section className="relative bg-white pt-24 overflow-hidden">
      {/* Structural Grid Background (Dify-inspired) */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="max-w-7xl mx-auto h-full border-x border-obs-border/50 relative">
          <div className="absolute top-0 left-0 w-full h-px bg-obs-border/30" />
          <div className="absolute top-[400px] left-0 w-full h-px bg-obs-border/30" />
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 md:px-10 relative">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-16 items-center py-20 md:py-32">
          
          {/* Left Content */}
          <div className="lg:col-span-7 flex flex-col items-start text-left stagger-children">
            {/* Announcement Banner (Dify style) */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-obs-border bg-surface mb-8 animate-rise opacity-0">
              <span className="flex h-2 w-2 rounded-full bg-amber animate-pulse" />
              <span className="text-[10px] font-mono font-bold uppercase tracking-[0.2em] text-amber">
                Veritas Protocol v1.0 is now live
              </span>
              <span className="material-symbols-outlined text-[14px] text-text-muted">north_east</span>
            </div>

            <h1 className="font-hero mb-8 text-text leading-[0.9] tracking-[-0.03em] animate-rise opacity-0 [animation-delay:80ms]">
              Infrastructure for the <br /> 
              <span className="text-amber italic">Consensus of Truth.</span>
            </h1>

            <p className="text-[18px] md:text-[20px] text-text-secondary leading-relaxed mb-12 max-w-xl animate-rise opacity-0 [animation-delay:160ms]">
              Verify claims with a hierarchical multi-agent intelligence layer. Veritas provides glass-box transparency, academic grounding, and web-scale evidence for the age of synthetic information.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-5 animate-rise opacity-0 [animation-delay:240ms]">
              <button
                onClick={() => router.push("/dashboard")}
                className="obs-btn bg-amber hover:bg-amber-hover text-white px-10 py-5 rounded-lg text-sm font-bold uppercase tracking-[0.2em] transition-all shadow-lg shadow-amber/10 group active:scale-[0.98]"
              >
                Initialize Audit
                <span className="material-symbols-outlined transition-transform group-hover:translate-x-1">arrow_forward</span>
              </button>
              <a
                href="#process"
                className="text-[12px] font-mono font-bold text-text-muted hover:text-amber uppercase tracking-[0.2em] transition-colors border-b border-obs-border pb-1"
              >
                View Protocol Overview
              </a>
            </div>
          </div>

          {/* Right Visual (Consensus Map) */}
          <div className="lg:col-span-5 relative animate-rise opacity-0 [animation-delay:320ms]">
             <div className="relative aspect-square max-w-[500px] mx-auto">
                {/* Central Node */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 rounded-3xl bg-white border border-obs-border shadow-xl flex items-center justify-center z-10 animate-float">
                   <div className="text-center">
                     <span className="material-symbols-outlined text-amber text-4xl mb-1">verified</span>
                     <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-widest">Consensus</p>
                   </div>
                </div>

                {/* Orbiting Agent Nodes */}
                {[
                  { icon: "psychology", label: "Director", pos: "top-0 left-1/2 -translate-x-1/2", delay: "0s" },
                  { icon: "travel_explore", label: "Evidence", pos: "bottom-0 left-1/2 -translate-x-1/2", delay: "1.5s" },
                  { icon: "account_tree", label: "Manager", pos: "left-0 top-1/2 -translate-y-1/2", delay: "0.8s" },
                  { icon: "menu_book", label: "Academic", pos: "right-0 top-1/2 -translate-y-1/2", delay: "2.2s" },
                ].map((node, i) => (
                  <div key={i} className={`absolute ${node.pos} w-24 h-24 rounded-2xl bg-surface border border-obs-border shadow-sm flex flex-col items-center justify-center animate-float`} style={{ animationDelay: node.delay }}>
                    <span className="material-symbols-outlined text-text-secondary text-xl mb-1">{node.icon}</span>
                    <span className="text-[9px] font-mono font-bold text-text-muted uppercase tracking-widest">{node.label}</span>
                  </div>
                ))}

                {/* Connecting Lines (SVG) */}
                <svg className="absolute inset-0 w-full h-full opacity-20 pointer-events-none" viewBox="0 0 500 500">
                  <path d="M250 250 L250 50" stroke="currentColor" strokeDasharray="4 4" className="text-obs-border" />
                  <path d="M250 250 L250 450" stroke="currentColor" strokeDasharray="4 4" className="text-obs-border" />
                  <path d="M250 250 L50 250" stroke="currentColor" strokeDasharray="4 4" className="text-obs-border" />
                  <path d="M250 250 L450 250" stroke="currentColor" strokeDasharray="4 4" className="text-obs-border" />
                </svg>
             </div>
          </div>
        </div>

        {/* Logo Ticker (Cohere / Dify style) */}
        <div className="pt-12 pb-24 border-t border-obs-border">
          <p className="text-[10px] font-mono font-bold text-text-muted uppercase tracking-[0.3em] mb-10 text-center opacity-60">Verified Signal Sources & Partners</p>
          <div className="relative overflow-hidden group">
            <div className="flex animate-marquee gap-16 items-center whitespace-nowrap min-w-full">
              {[...logos, ...logos].map((logo, i) => (
                <div key={i} className="flex items-center gap-4 grayscale opacity-80 hover:grayscale-0 hover:opacity-100 transition-all cursor-default pr-20">
                  <span className="material-symbols-outlined text-2xl">{logo.icon}</span>
                  <span className="text-sm font-display font-medium text-text">{logo.name}</span>
                </div>
              ))}
            </div>
            {/* Fades for smooth edges */}
            <div className="absolute inset-y-0 left-0 w-12 bg-gradient-to-r from-white via-white/50 to-transparent z-10" />
            <div className="absolute inset-y-0 right-0 w-12 bg-gradient-to-l from-white via-white/50 to-transparent z-10" />
          </div>
        </div>
      </div>
    </section>
  );
}
