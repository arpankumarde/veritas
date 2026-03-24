"use client";

import LandingNav from "@/components/landing/nav";
import LandingHero from "@/components/landing/hero";
import LandingProcess from "@/components/landing/process";
import LandingFooter from "@/components/landing/footer";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white text-text selection:bg-amber/20 selection:text-amber overflow-x-hidden">
      <LandingNav />
      
      <main>
        <LandingHero />
        
        {/* Feature Highlights Section (Dify-inspired) */}
        <section id="features" className="py-24 px-6 bg-surface border-t border-b border-obs-border">
          <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
            <div className="stagger-children">
              <h4 className="text-[11px] font-mono font-bold uppercase tracking-[0.2em] text-amber mb-6 animate-rise opacity-0">Consensus Engine</h4>
              <h2 className="font-section mb-8 text-text animate-rise opacity-0 [animation-delay:80ms]">Hierarchical Multi-Agent Architecture</h2>
              <p className="font-sub mb-10 text-text-secondary leading-relaxed animate-rise opacity-0 [animation-delay:160ms]">
                Unlike simple search wrappers, Veritas employs a tiered approach to truth. A Director agent plans the audit strategy, Manager agents verify sub-claims, and a pool of Intern agents gather evidence from high-signal sources.
              </p>
              
              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-8 animate-rise opacity-0 [animation-delay:240ms]">
                <li className="flex flex-col gap-4 p-6 border border-obs-border bg-white rounded-lg">
                  <div className="w-8 h-8 rounded bg-emerald-soft border border-emerald/10 flex items-center justify-center">
                    <span className="material-symbols-outlined text-[14px] text-emerald font-bold">menu_book</span>
                  </div>
                  <div>
                    <h5 className="text-sm font-semibold text-text mb-1 uppercase tracking-tight">Academic Audit</h5>
                    <p className="text-[13px] text-text-secondary leading-relaxed">Cross-references claims with Semantic Scholar and peer-reviewed arXiv repositories.</p>
                  </div>
                </li>
                <li className="flex flex-col gap-4 p-6 border border-obs-border bg-white rounded-lg">
                  <div className="w-8 h-8 rounded bg-amber-soft border border-amber/10 flex items-center justify-center">
                    <span className="material-symbols-outlined text-[14px] text-amber font-bold">public</span>
                  </div>
                  <div>
                    <h5 className="text-sm font-semibold text-text mb-1 uppercase tracking-tight">Global Signal</h5>
                    <p className="text-[13px] text-text-secondary leading-relaxed">Gathering live evidence with Bright Data SERP unlocking and real-time indexing.</p>
                  </div>
                </li>
              </ul>
            </div>

            <div className="relative animate-rise opacity-0 [animation-delay:320ms]">
              <div className="relative obs-card bg-white p-1 border-obs-border shadow-md rounded-lg overflow-hidden">
                <div className="bg-surface rounded-lg p-6 font-mono text-[11px] text-text-secondary overflow-hidden h-[400px]">
                  <div className="mb-4 flex items-center gap-2 text-text border-b border-obs-border pb-4">
                    <span className="w-2 h-2 rounded-full bg-slate-300" />
                    <span className="w-2 h-2 rounded-full bg-slate-300" />
                    <span className="w-2 h-2 rounded-full bg-slate-300" />
                    <span className="ml-2 text-[10px] font-bold uppercase tracking-widest text-text-muted">system_executive.log</span>
                  </div>
                  <div className="space-y-3 opacity-90 overflow-y-auto h-full pr-2">
                    <p className="text-amber font-bold">[DIRECTOR] Decomposing claim ID: 0xFF12</p>
                    <p className="text-text-secondary ml-4">→ Analyzing claim: "Mars was once habitable..."</p>
                    <p className="text-text-secondary ml-4">→ Breaking into sub-claims (Habitability, Liquid Water, Methane)</p>
                    <p className="text-cyan font-bold">[MANAGER] Distributing sub-claims to Intern Pool</p>
                    <p className="text-text-secondary ml-4">[INTERN_04] Tasked: Academic Search (Scholar)</p>
                    <p className="text-text-secondary ml-4">[INTERN_09] Tasked: Web Index (SERP)</p>
                    <p className="text-cyan font-bold">[INTERN_04] Retrieval complete. 12 papers indexed.</p>
                    <p className="text-cyan font-bold">[INTERN_09] Retrieval complete. 42 sources indexed.</p>
                    <p className="text-amber font-bold">[MANAGER] Consensus weighting started...</p>
                    <p className="text-emerald font-bold">[MANAGER] Verdict determined: SUPPORTIVE (92% Confidence)</p>
                    <p className="text-amber font-bold">[DIRECTOR] Generating consensus report.</p>
                    <div className="animate-pulse flex gap-1 h-2 mt-4">
                       <span className="w-2 bg-amber" />
                       <span className="w-1 bg-amber/50" />
                       <span className="w-1 bg-amber/20" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Serious CTA Section */}
        <section className="py-32 px-6 flex flex-col items-center text-center max-w-7xl mx-auto">
          <div className="max-w-2xl stagger-children flex flex-col items-center">
             <h2 className="font-section mb-6 text-text uppercase tracking-tight">Access the Observatory</h2>
             <p className="font-sub mb-10 text-text-secondary leading-relaxed">
               Secure the integrity of your information environment. Launch the Fact Check command center and verify claims with hierarchical multi-agent logic.
             </p>
             <button
               onClick={() => window.location.href = "/dashboard"}
               className="obs-btn btn-primary px-12 py-4 text-[15px] font-bold rounded-lg bg-amber hover:bg-amber-hover text-white transition-all shadow-none uppercase tracking-widest"
             >
               Launch Dashboard
             </button>
          </div>
        </section>
      </main>

      <LandingFooter />
    </div>
  );
}
