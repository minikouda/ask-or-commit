"""
Generate all report figures from results/final_experiment/experiment_summary.json
and results/final_experiment/experiment_records.jsonl.

Output: report/figures/fig{1,2,3}_*.{pdf,png}

Usage:
    python scripts/plot_figures.py
"""

import json
import os
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SUMMARY_PATH = "results/final_experiment/experiment_summary.json"
RECORDS_PATH = "results/final_experiment/experiment_records.jsonl"
OUT_DIR = "report/figures"

os.makedirs(OUT_DIR, exist_ok=True)

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

with open(RECORDS_PATH) as f:
    records = [json.loads(l) for l in f if l.strip()]

# ── shared label maps ────────────────────────────────────────────────────────

SPEAKER_LABELS = {
    "strategic-natural(gemini-2.0-flash-001)":       "natural",
    "strategic-listener_aware(gemini-2.0-flash-001)": "listener-aware",
    "strategic-contrastive(gemini-2.0-flash-001)":   "contrastive",
    "strategic-scene_first(gemini-2.0-flash-001)":   "scene-first",
    "feature-canonical":                              "canonical (rule)",
    "strategic-pragmatic(gemini-2.0-flash-001)":     "pragmatic",
    "strategic-superlative(gemini-2.0-flash-001)":   "superlative",
    "strategic-landmark(gemini-2.0-flash-001)":      "landmark",
}
SP_COLOR = {
    "natural":        "#1f77b4",
    "listener-aware": "#2ca02c",
    "contrastive":    "#ff7f0e",
    "scene-first":    "#9467bd",
    "canonical (rule)": "#8c564b",
    "pragmatic":      "#e377c2",
    "superlative":    "#bcbd22",
    "landmark":       "#d62728",
}
SP_STYLE = {
    "natural": "-o", "listener-aware": "-s", "contrastive": "-^",
    "scene-first": "-D", "canonical (rule)": "--o", "pragmatic": "-v",
    "superlative": "-x", "landmark": "-*",
}

LISTENER_LABELS = {
    "index(gemini-2.0-flash-001)": "index",
    "direct(gemini-2.0-flash-001)": "direct",
    "elimination(gemini-2.0-flash-001)": "elimination",
    "feature-match(gemini-2.0-flash-001)": "feat-match",
    "vllm-listener(gemini-2.0-flash-001,σ=10.0)": "coordinate",  # noqa: RUF001
    "cot(gemini-2.0-flash-001)": "cot",
}
LS_COLOR = {
    "index": "#1f77b4", "direct": "#ff7f0e",
    "elimination": "#9467bd", "feat-match": "#e377c2",
    "coordinate": "#d62728", "cot": "#8c564b",
}
LS_MARKER = {
    "index": "o", "direct": "^", "elimination": "D",
    "feat-match": "P", "coordinate": "*", "cot": "X",
}
LS_SIZE = {6: 60, 8: 110, 10: 180}


# ── Figure 1: Speaker accuracy vs. scene size ────────────────────────────────

def plot_fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ci, cond in enumerate(["none", "force"]):
        ax = axes[ci]
        subset = [d for d in summary
                  if d["listener_type"] == "index(gemini-2.0-flash-001)"
                  and d["condition"] == cond]

        for sp_key, label in SPEAKER_LABELS.items():
            rows = sorted([d for d in subset if d["speaker_type"] == sp_key],
                          key=lambda x: x["scene_size"])
            sizes = [r["scene_size"] for r in rows]
            accs  = [r["accuracy"]   for r in rows]
            ax.plot(sizes, accs, SP_STYLE[label], color=SP_COLOR[label],
                    label=label, linewidth=1.6, markersize=6)

        ax.set_xlabel("Scene size $N$", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            f"({'ab'[ci]}) {'No forced' if cond=='none' else 'Forced same-color'} overlap "
            f"($\mathrm{{{cond}}}$)", fontsize=11)
        ax.set_xticks([6, 8, 10])
        ax.set_ylim(0.60 if cond == "force" else 0.78, 1.02)
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        kw = {"bbox_inches": "tight"} if ext == "pdf" else {"dpi": 150, "bbox_inches": "tight"}
        plt.savefig(os.path.join(OUT_DIR, f"fig1_speaker_scaling.{ext}"), **kw)
    plt.close()
    print("Figure 1 saved.")


# ── Figure 2: Listener scatter (accuracy vs. ask rate) ──────────────────────

def plot_fig2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs_iso = np.linspace(0, 0.9, 200)

    for ci, cond in enumerate(["none", "force"]):
        ax = axes[ci]
        fc = [d for d in summary
              if d["speaker_type"] == "feature-canonical"
              and d["condition"] == cond]

        seen = set()
        for d in fc:
            label = LISTENER_LABELS.get(d["listener_type"])
            if label is None:
                continue
            lbl = label if label not in seen else "_nolegend_"
            seen.add(label)
            ax.scatter(d["clarification_rate"], d["accuracy"],
                       s=LS_SIZE[d["scene_size"]], color=LS_COLOR[label],
                       marker=LS_MARKER[label], label=lbl,
                       alpha=0.85, edgecolors="k", linewidths=0.4)

        for cpa_val, ls in [(0.95, ":"), (0.9, "--"), (0.8, "-.")]:
            ys = cpa_val + 0.25 * xs_iso
            ax.plot(xs_iso, ys, color="gray", linestyle=ls, linewidth=1,
                    label=f"CPA={cpa_val}" if ci == 0 else "_nolegend_")

        ax.set_xlabel("Clarification rate (ask%)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlim(-0.02, 0.85)
        ax.set_ylim(0.45, 1.05)
        ax.set_title(
            f"({'ab'[ci]}) Condition: $\mathrm{{{cond}}}$", fontsize=11)
        ax.grid(alpha=0.3)

        if ci == 0:
            for sz in [6, 8, 10]:
                ax.scatter([], [], s=LS_SIZE[sz], color="gray",
                           alpha=0.7, label=f"$N={sz}$")
        if ci == 1:
            ax.legend(fontsize=8, loc="lower left", ncol=2)
        else:
            ax.legend(fontsize=8, loc="lower right", ncol=2)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        kw = {"bbox_inches": "tight"} if ext == "pdf" else {"dpi": 150, "bbox_inches": "tight"}
        plt.savefig(os.path.join(OUT_DIR, f"fig2_listener_scatter.{ext}"), **kw)
    plt.close()
    print("Figure 2 saved.")


# ── Figure 3: EU calibration (entropy | calibration curve | CPA heatmap) ────

def plot_fig3():
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    text_recs = [r for r in records if "vllm-listener" not in r["listener_type"]]

    # Panel A: entropy histogram
    ax1 = fig.add_subplot(gs[0])
    ask_ent    = [min(r["entropy"], 3) for r in text_recs
                  if r["action"] == "ask" and r.get("entropy") is not None]
    commit_ent = [min(r["entropy"], 3) for r in text_recs
                  if r["action"] == "commit" and r.get("entropy") is not None]
    bins = np.linspace(0, 3, 31)
    ax1.hist(commit_ent, bins=bins, color="#2196F3", alpha=0.7,
             label="commit", density=True)
    ax1.hist(ask_ent,    bins=bins, color="#F44336", alpha=0.7,
             label="ask",    density=True)
    ax1.axvline(0.415, color="k", linestyle="--", linewidth=1.5,
                label="EU threshold")
    ax1.set_xlabel("Referential entropy $H(T|u)$ (bits)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("(a) Entropy by action", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel B: calibration curve
    ax2 = fig.add_subplot(gs[1])
    eu_edges = [0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.001]
    bin_correct = collections.defaultdict(list)
    for r in text_recs:
        if r["action"] != "commit":
            continue
        eu = r.get("eu_commit") or 0.0
        for i in range(len(eu_edges) - 1):
            if eu_edges[i] <= eu < eu_edges[i + 1]:
                bin_correct[i].append(int(r["correct"]))
                break

    xs, ys, ns = [], [], []
    for i in range(len(eu_edges) - 1):
        if len(bin_correct[i]) > 10:
            xs.append((eu_edges[i] + eu_edges[i + 1]) / 2)
            ys.append(np.mean(bin_correct[i]))
            ns.append(len(bin_correct[i]))

    ax2.scatter(xs, ys, s=[max(20, n / 20) for n in ns],
                color="#2ca02c", zorder=3, edgecolors="k", linewidths=0.5)
    ax2.plot(xs, ys, color="#2ca02c", linewidth=1.5, zorder=2)
    ax2.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1,
             label="perfect calibration")
    ax2.axvline(0.75, color="gray", linestyle=":", linewidth=1.5,
                label="EU threshold ($c{=}0.25$)")
    ax2.set_xlabel(r"Listener confidence $\max_i P(t_i|u)$", fontsize=11)
    ax2.set_ylabel("Empirical accuracy", fontsize=11)
    ax2.set_title("(b) Calibration on committed scenes", fontsize=11)
    ax2.set_xlim(0.45, 1.02)
    ax2.set_ylim(0.45, 1.02)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)

    # Panel C: CPA heatmap
    ax3 = fig.add_subplot(gs[2])

    sp_order = ["natural", "listener-aware", "contrastive", "scene-first",
                "canonical", "pragmatic", "superlative", "landmark"]
    ls_order = ["index", "direct", "feat-match",
                "elimination", "coordinate", "cot"]

    sp_map = {v: k for k, v in SPEAKER_LABELS.items()}
    sp_map["canonical"] = "feature-canonical"

    heatmap = np.full((len(ls_order), len(sp_order)), np.nan)
    s8_none = [d for d in summary
               if d["scene_size"] == 8 and d["condition"] == "none"]
    for d in s8_none:
        sp = SPEAKER_LABELS.get(d["speaker_type"])
        ls = LISTENER_LABELS.get(d["listener_type"])
        if sp and ls:
            sp_short = sp if sp != "canonical (rule)" else "canonical"
            if sp_short in sp_order and ls in ls_order:
                heatmap[ls_order.index(ls), sp_order.index(sp_short)] = d["cpa"]

    im = ax3.imshow(heatmap, cmap="RdYlGn", vmin=0.1, vmax=1.0, aspect="auto")
    ax3.set_xticks(range(len(sp_order)))
    ax3.set_xticklabels(sp_order, rotation=40, ha="right", fontsize=8)
    ax3.set_yticks(range(len(ls_order)))
    ax3.set_yticklabels(ls_order, fontsize=9)
    ax3.set_title(r"(c) CPA heatmap ($N{=}8$, $\mathrm{none}$)", fontsize=11)
    for i in range(len(ls_order)):
        for j in range(len(sp_order)):
            if not np.isnan(heatmap[i, j]):
                val = heatmap[i, j]
                ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7, color="white" if val < 0.5 else "black")
    plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.03)

    plt.savefig(os.path.join(OUT_DIR, "fig3_eu_calibration.pdf"),
                bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig3_eu_calibration.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 3 saved.")


# ── Figure 4: Overlap analysis ───────────────────────────────────────────────

def plot_fig4():
    LS_ORDER_4 = ["index", "direct", "feat-match",
                  "elimination", "coordinate", "cot"]
    LS_KEY_4 = {
        "index":       "index(gemini-2.0-flash-001)",
        "direct":      "direct(gemini-2.0-flash-001)",
        "feat-match":  "feature-match(gemini-2.0-flash-001)",
        "elimination": "elimination(gemini-2.0-flash-001)",
        "coordinate":  "vllm-listener(gemini-2.0-flash-001,σ=10.0)",
        "cot":         "cot(gemini-2.0-flash-001)",
    }
    LS_COLOR_4 = ["#1f77b4", "#ff7f0e", "#e377c2",
                  "#9467bd", "#d62728", "#8c564b"]
    SP_KEYS_4 = {
        "natural":     "strategic-natural(gemini-2.0-flash-001)",
        "l-aware":     "strategic-listener_aware(gemini-2.0-flash-001)",
        "contrastive": "strategic-contrastive(gemini-2.0-flash-001)",
        "scene-1st":   "strategic-scene_first(gemini-2.0-flash-001)",
        "canonical":   "feature-canonical",
        "pragmatic":   "strategic-pragmatic(gemini-2.0-flash-001)",
        "superlative": "strategic-superlative(gemini-2.0-flash-001)",
        "landmark":    "strategic-landmark(gemini-2.0-flash-001)",
    }
    SP_ORDER_4 = list(SP_KEYS_4.keys())

    fig = plt.figure(figsize=(14, 4.5))
    gs4 = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # Panel A: mean CPA none vs force per listener
    ax1 = fig.add_subplot(gs4[0])
    none_cpas, force_cpas = [], []
    for ls in LS_ORDER_4:
        lsk = LS_KEY_4[ls]
        none_cpas.append(np.mean([d["cpa"] for d in summary
                                  if d["listener_type"] == lsk and d["condition"] == "none"]))
        force_cpas.append(np.mean([d["cpa"] for d in summary
                                   if d["listener_type"] == lsk and d["condition"] == "force"]))
    x = np.arange(len(LS_ORDER_4))
    w = 0.35
    ax1.bar(x - w/2, none_cpas,  w, label="none",  color=[c + "cc" for c in LS_COLOR_4],
            edgecolor="k", linewidth=0.6)
    ax1.bar(x + w/2, force_cpas, w, label="force", color=LS_COLOR_4,
            edgecolor="k", linewidth=0.6, hatch="//")
    ax1.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(LS_ORDER_4, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Mean CPA (across all speakers, $N$)", fontsize=10)
    ax1.set_title("(a) CPA: none vs.\ force, per listener", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: ask rate delta per listener per scene size
    ax2 = fig.add_subplot(gs4[1])
    colors_sz = ["#aec6e8", "#5fa5d8", "#1f77b4"]
    x2 = np.arange(len(LS_ORDER_4))
    offsets = [-0.27, 0, 0.27]
    for si, (sz, col) in enumerate(zip([6, 8, 10], colors_sz)):
        deltas = []
        for ls in LS_ORDER_4:
            lsk = LS_KEY_4[ls]
            n_ask = np.mean([d["clarification_rate"] for d in summary
                             if d["listener_type"] == lsk and d["condition"] == "none"
                             and d["scene_size"] == sz])
            f_ask = np.mean([d["clarification_rate"] for d in summary
                             if d["listener_type"] == lsk and d["condition"] == "force"
                             and d["scene_size"] == sz])
            deltas.append(f_ask - n_ask)
        ax2.bar(x2 + offsets[si], deltas, 0.25, label=f"$N={sz}$",
                color=col, edgecolor="k", linewidth=0.5)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(LS_ORDER_4, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel(r"$\Delta$ ask rate (force $-$ none)", fontsize=10)
    ax2.set_title("(b) Ask rate increase under forced overlap", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # Panel C: CPA delta heatmap at N=8
    ax3 = fig.add_subplot(gs4[2])
    heatmap4 = np.full((len(LS_ORDER_4), len(SP_ORDER_4)), np.nan)
    for d in summary:
        if d["scene_size"] != 8:
            continue
        ls = next((k for k, v in LS_KEY_4.items() if v == d["listener_type"]), None)
        sp = next((k for k, v in SP_KEYS_4.items() if v == d["speaker_type"]), None)
        if ls and sp:
            i, j = LS_ORDER_4.index(ls), SP_ORDER_4.index(sp)
            if np.isnan(heatmap4[i, j]):
                heatmap4[i, j] = 0.0
            heatmap4[i, j] += d["cpa"] if d["condition"] == "force" else -d["cpa"]
    im = ax3.imshow(heatmap4, cmap="RdYlGn", vmin=-0.8, vmax=0.1, aspect="auto")
    ax3.set_xticks(range(len(SP_ORDER_4)))
    ax3.set_xticklabels(SP_ORDER_4, rotation=40, ha="right", fontsize=8)
    ax3.set_yticks(range(len(LS_ORDER_4)))
    ax3.set_yticklabels(LS_ORDER_4, fontsize=9)
    ax3.set_title(r"(c) $\Delta$ CPA = force $-$ none ($N{=}8$)", fontsize=11)
    for i in range(len(LS_ORDER_4)):
        for j in range(len(SP_ORDER_4)):
            if not np.isnan(heatmap4[i, j]):
                v = heatmap4[i, j]
                ax3.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=7, color="white" if v < -0.4 else "black")
    plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.03)

    plt.savefig(os.path.join(OUT_DIR, "fig4_overlap_analysis.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig4_overlap_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 4 saved.")


if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
