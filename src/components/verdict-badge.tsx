import { cn } from "@/lib/utils";

const verdictConfig: Record<
  string,
  { label: string; badgeClass: string; className: string }
> = {
  true: {
    label: "True",
    badgeClass: "badge-success",
    className: "bg-emerald-500/15 text-emerald-400 ring-emerald-500/30",
  },
  mostly_true: {
    label: "Mostly True",
    badgeClass: "badge-success",
    className: "bg-emerald-500/10 text-emerald-300 ring-emerald-500/20",
  },
  mixed: {
    label: "Mixed",
    badgeClass: "badge-warning",
    className: "bg-amber-500/15 text-amber-400 ring-amber-500/30",
  },
  mostly_false: {
    label: "Mostly False",
    badgeClass: "badge-error",
    className: "bg-orange-500/15 text-orange-400 ring-orange-500/30",
  },
  false: {
    label: "False",
    badgeClass: "badge-error",
    className: "bg-red-500/15 text-red-400 ring-red-500/30",
  },
  unverified: {
    label: "Unverified",
    badgeClass: "badge-system",
    className: "bg-zinc-500/15 text-zinc-400 ring-zinc-500/30",
  },
  insufficient_evidence: {
    label: "Insufficient Evidence",
    badgeClass: "badge-system",
    className: "bg-zinc-500/15 text-zinc-400 ring-zinc-500/30",
  },
};

function normalizeVerdict(verdict: string): string {
  return verdict.toLowerCase().replace(/[\s-]+/g, "_");
}

interface VerdictBadgeProps {
  verdict?: string;
  size?: "sm" | "default" | "lg";
  className?: string;
}

export function VerdictBadge({
  verdict,
  size = "default",
  className,
}: VerdictBadgeProps) {
  if (!verdict) {
    return (
      <span
        className={cn(
          "inline-flex items-center rounded-md font-medium ring-1 ring-inset",
          "bg-zinc-500/15 text-zinc-400 ring-zinc-500/30",
          size === "sm" && "px-2 py-0.5 text-xs",
          size === "default" && "px-2.5 py-1 text-xs",
          size === "lg" && "px-3.5 py-1.5 text-sm",
          className
        )}
      >
        Pending
      </span>
    );
  }

  const key = normalizeVerdict(verdict);
  const config = verdictConfig[key] ?? {
    label: verdict,
    badgeClass: "badge-system",
    className: "bg-zinc-500/15 text-zinc-400 ring-zinc-500/30",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md font-medium ring-1 ring-inset",
        config.className,
        size === "sm" && "px-2 py-0.5 text-xs",
        size === "default" && "px-2.5 py-1 text-xs",
        size === "lg" && "px-3.5 py-1.5 text-sm",
        className
      )}
    >
      {config.label}
    </span>
  );
}
