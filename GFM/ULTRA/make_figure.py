"""Summary figure from <results>/summary.csv -> <results>/figure.{pdf,png}.

Two panels sharing one row per information-preserving transformation (grouped by dataset): left, Kendall tau between ULTRA's
rankings on the original and the transformed graph (dashed line = repeat run of the same graph, the noise
floor); right, change in MRR. One marker per checkpoint: colour (validated categorical slots 1-3) plus a
distinct marker shape, so it also reads in grayscale print. Graphs held out from a checkpoint's pretraining
(zero-shot) use open markers.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS", os.path.join(HERE, "results"))
CKPT_STYLE = {"ultra_3g": ("#2a78d6", "o", "ULTRA 3g"), "ultra_4g": ("#eb6834", "s", "ULTRA 4g"),
              "ultra_50g": ("#1baf7a", "D", "ULTRA 50g")}
INK, QUIET, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DS_NAME = {"WN18RR": "WN18RR", "FB15k237": "FB15k-237", "Hetionet": "Hetionet (biomedical)"}
VAR_NAME = {
    "all_inverse_edges": "+ named inverse edges", "inverse": "+ named inverse edges",
    "transitive_edges": "+ 2/3-hop shortcuts", "universal_edges": "union of all representations",
    "reified_cvt_light": "Freebase mediator nodes restored (light)",
    "reified_cvt_strong": "Freebase mediator nodes restored (strong)",
    "reified_regulation": "regulation edges as event nodes",
    "shortcut2_c5k": "+ 2-hop shortcuts, 5K per type", "shortcut2_c50k": "+ 2-hop shortcuts, 50K per type",
    "shortcut2_c200k": "+ 2-hop shortcuts, 200K per type",
}


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, "summary.csv"))))
    rank = {v: i for i, v in enumerate(VAR_NAME)}  # VAR_NAME order = reading order (dose-response ascending)
    order = sorted({(r["dataset"], r["variant"]) for r in rows},
                   key=lambda k: ({"WN18RR": 0, "FB15k237": 1, "Hetionet": 2}.get(k[0], 9), rank.get(k[1], 99)))
    # y positions with a gap and a header between datasets
    ypos, labels, headers, y = {}, [], [], 0
    last_ds = None
    for ds, var in order:
        if ds != last_ds:
            if last_ds is not None:
                y += 0.6
            headers.append((y, DS_NAME.get(ds, ds)))
            y += 1
            last_ds = ds
        ypos[(ds, var)] = y
        labels.append((y, VAR_NAME.get(var, var)))
        y += 1
    taus = [float(r["kendall_tau"]) for r in rows]
    floors = {float(r["noise_floor_tau"]) for r in rows if r["noise_floor_tau"] not in ("", "None")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 0.25 * y + 1.25), sharey=True,
                                   gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.16})
    off = {"ultra_3g": -0.2, "ultra_4g": 0.0, "ultra_50g": 0.2}
    for r in rows:
        c, m, lab = CKPT_STYLE[r["checkpoint"]]
        yy = ypos[(r["dataset"], r["variant"])] + off[r["checkpoint"]]
        filled = r["held_out"] == "yes"
        kw = dict(color=c, marker=m, markersize=5.5, linestyle="none", markeredgewidth=1.4,
                  markerfacecolor=c if not filled else "white")
        ax1.plot(float(r["kendall_tau"]), yy, **kw)
        ax2.plot(float(r["delta_mrr"]), yy, **kw)
    floor = min(floors) if floors else 1.0
    ax1.axvline(floor, color=QUIET, linestyle=(0, (4, 3)), linewidth=1.2)
    ax1.text(floor - 0.01, -1.75, f"same graph twice, or transformation declared", ha="right", va="center",
             fontsize=7.5, color=QUIET)
    ax1.text(floor - 0.01, -1.25, f"and compiled back (no retraining): \u03c4 = {floor:.3f}", ha="right", va="center",
             fontsize=7.5, color=QUIET)
    ax2.axvline(0, color=QUIET, linewidth=1.0)
    for ax in (ax1, ax2):
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(QUIET)
        ax.tick_params(colors=QUIET, labelsize=8.5, left=False)
    ax1.set_xlim(min(0, min(taus) - 0.05), 1.03)
    ax1.set_xlabel("Kendall \u03c4 vs. original graph (1 = identical)", fontsize=8, color=INK)
    lo = min(float(r["delta_mrr"]) for r in rows)
    hi = max(float(r["delta_mrr"]) for r in rows)
    ax2.set_xlim(lo - 0.015, max(hi, 0) + 0.03)
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.set_xlabel("Change in MRR", fontsize=8, color=INK)
    ax1.set_yticks([p for p, _ in labels])
    ax1.set_yticklabels([t for _, t in labels], fontsize=7.5, color=INK)
    for p, t in headers:
        ax1.text(-0.02, p, t, transform=ax1.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                 fontweight="bold", color=INK)
    ax1.set_ylim(y - 0.4, -2.2)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=5.5, label=lab)
               for c, m, lab in CKPT_STYLE.values()]
    handles.append(plt.Line2D([], [], color=QUIET, marker="o", markerfacecolor="white", linestyle="none",
                              markersize=5.5, label="graph not in pretraining (documented)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.58, -0.01))
    fig.suptitle(f"Representing the same facts differently changes ULTRA's rankings "
                 f"(\u03c4 {min(taus):.2f}\u2013{max(taus):.2f})", x=0.01, ha="left", fontsize=9.5,
                 fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.37, right=0.98, top=1 - 0.55 / fig.get_figheight(), bottom=0.85 / fig.get_figheight())
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"figure.{ext}"), dpi=200)
    print("wrote", os.path.join(RES, "figure.{pdf,png}"))


if __name__ == "__main__":
    main()
