"use client";

export default function LandingProcess() {
  const steps = [
    {
      title: "Decompose",
      description: "Director agents break complex claims into atoms of verifiable information.",
      icon: "account_tree",
      color: "text-amber"
    },
    {
      title: "Investigate",
      description: "Parallel intern pools scour web and academic journals for evidence.",
      icon: "travel_explore",
      color: "text-cyan"
    },
    {
      title: "Reason",
      description: "Manager agents weigh evidence quality and consensus to determine truth.",
      icon: "psychology",
      color: "text-violet"
    },
    {
      title: "Verdict",
      description: "A final determination with citations, reasoning, and uncertainty metrics.",
      icon: "verified",
      color: "text-emerald"
    }
  ];

  return (
    <section id="process" className="py-24 px-6 max-w-7xl mx-auto border-t border-b border-obs-border">
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-16 gap-8">
        <div className="max-w-xl">
          <h2 className="font-section mb-6 text-text">The Verification Pipeline</h2>
          <p className="text-text-secondary text-base leading-relaxed">
            Built for precision, scalability, and radical transparency. Our hierarchical agent architecture ensures every claim is thoroughly audited.
          </p>
        </div>
        <div className="text-[11px] font-mono font-bold uppercase tracking-widest text-text-muted border-l border-amber pl-6 py-2">
          Protocol 01-A <br />
          Multi-Agent Consensus
        </div>
      </div>

      <div className="grid-container grid-cols-1 md:grid-cols-2 lg:grid-cols-4 bg-white border-obs-border">
        {steps.map((step, i) => (
          <div key={i} className="grid-cell group hover:bg-surface transition-colors cursor-default border-obs-border">
            <div className={`mb-8 p-3 rounded-lg bg-amber-soft border border-amber/10 inline-flex group-hover:scale-105 transition-transform duration-300`}>
              <span className={`material-symbols-outlined text-2xl ${step.color}`}>
                {step.icon}
              </span>
            </div>
            <h3 className="text-lg font-display font-semibold mb-4 text-text uppercase tracking-tight">
              {step.title}
            </h3>
            <p className="text-sm text-text-secondary leading-relaxed">
              {step.description}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
