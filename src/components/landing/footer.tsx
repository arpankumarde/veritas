"use client";

export default function LandingFooter() {
  return (
    <footer className="px-6 py-20 bg-white border-t border-obs-border">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-12 md:gap-8">
        <div className="col-span-1 md:col-span-2">
          <div className="flex items-center gap-3 mb-6">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-amber-soft border border-amber/10">
              <span className="material-symbols-outlined text-amber text-lg">verified</span>
            </div>
            <h1 className="text-xl font-display font-semibold tracking-tight text-text uppercase">Veritas</h1>
          </div>
          <p className="text-sm text-text-secondary max-w-sm mb-8 leading-relaxed">
            Infrastructure for automated truth verification in an asymmetric information environment. Multi-agent consensus protocol v1.0.
          </p>
          <div className="flex items-center gap-4">
            <a href="#" className="w-9 h-9 border border-obs-border flex items-center justify-center hover:bg-surface transition-colors">
              <span className="material-symbols-outlined text-sm">share</span>
            </a>
            <a href="#" className="w-9 h-9 border border-obs-border flex items-center justify-center hover:bg-surface transition-colors">
              <span className="material-symbols-outlined text-sm">code</span>
            </a>
          </div>
        </div>

        <div>
          <h4 className="text-[11px] font-mono font-bold uppercase tracking-[0.2em] text-amber mb-6">System Architecture</h4>
          <ul className="space-y-4 text-[13px] text-text-secondary">
            <li><a href="#" className="hover:text-amber transition-colors">Director Protocol</a></li>
            <li><a href="#" className="hover:text-amber transition-colors">Evidence Knowledge Graph</a></li>
            <li><a href="#" className="hover:text-amber transition-colors">API Specifications</a></li>
            <li><a href="#" className="hover:text-amber transition-colors">Peer-Reviewed Methodology</a></li>
          </ul>
        </div>

        <div>
          <h4 className="text-[11px] font-mono font-bold uppercase tracking-[0.2em] text-amber mb-6">Governance</h4>
          <ul className="space-y-4 text-[13px] text-text-secondary">
            <li><a href="#" className="hover:text-amber transition-colors">Transparency Report</a></li>
            <li><a href="#" className="hover:text-amber transition-colors">Data Privacy Policy</a></li>
            <li><a href="#" className="hover:text-amber transition-colors">System Status</a></li>
            <li><a href="#" className="hover:text-amber transition-colors">Contact Engineering</a></li>
          </ul>
        </div>
      </div>
      
      <div className="max-w-7xl mx-auto mt-20 pt-8 border-t border-obs-border flex flex-col md:flex-row justify-between items-center gap-6">
        <p className="text-[11px] font-mono text-text-muted">
          © {new Date().getFullYear()} VERITAS HIERARCHICAL SYSTEMS. 
        </p>
        <div className="flex items-center gap-6">
          <p className="text-[11px] font-mono text-text-muted uppercase tracking-wider">Status: <span className="text-emerald font-bold">OPERATIONAL</span></p>
          <p className="text-[11px] font-mono text-text-muted uppercase tracking-wider opacity-60">Verified Hash: 0x2A ... B3C</p>
        </div>
      </div>
    </footer>
  );
}
