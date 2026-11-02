"""Figure S5: the unsoundness window of SERIALIZABLE on an HNSW index (space-time diagram).

SCHEMATIC.  This figure is drawn from the mechanism, not from measurements: it contains no data.  The mechanism is the
one reproduced step by step by ariadne/pgtests/hole.py (A6_ssi_soundness, hole.jsonl: both transactions commit in
every HOLE trial, none in the CONTROL trials) and closed by patches/pgvector_ssi_recheck.patch:

    index_insert() calls CheckForSerializableConflictIn(index) BEFORE the access method's insert function;
    hnswinsert first searches the graph for the neighbours of the new element and only then links it.
    A reader that takes its relation-level SIREAD lock after the writer's check and scans before the writer's element
    is linked leaves no read-write dependency behind: both transactions commit, and no serial order explains both.

Shows     two transactions as swim lanes, time left to right.  T1 (writer): its own search, then INSERT = conflict
          check, graph search, link, and COMMIT.  T2 (reader, then writer): SIREAD lock on the whole index, k-NN scan
          that does not see T1's element, INSERT, COMMIT.  The shaded band is the window between T1's conflict check
          and the linking of its element; the marker after the link is where the patch repeats the check.
Paper     main text, figure "the window" (one column).
"""
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

from . import style as st, load as ld

NAME = "fig_s_05_ssi_window"

T1Y, T2Y = 0.70, 0.31          # lane positions (axes units)
X_CHECK, X_LINK = 28.0, 68.0   # T1: conflict check, element linked
X_PATCH = 74.0


def _box(ax, x0, x1, y, text, fc, ec, fs, h=0.115, color=None, z=3):
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.9", fc=fc, ec=ec, lw=0.6,
                                zorder=z, mutation_aspect=0.012))
    ax.text((x0 + x1) / 2, y - 0.003, text, ha="center", va="center", fontsize=fs, color=color or st.INK, zorder=z + 1)


def _tick(ax, x, y, color, h=0.07, lw=1.5, z=4):
    ax.plot([x, x], [y - h, y + h], color=color, lw=lw, solid_capstyle="butt", zorder=z)


def draw(ax, fs=6.0):
    """Draw the diagram into ax (the axis is switched off).  Coordinates: x in [0, 100] is time, y in [0, 1]."""
    ink, mut = st.INK, st.MUTED
    red = st.ACCENT; blue = st.OKABE_ITO["blue"]; green = st.OKABE_ITO["green"]
    fill = st.lighten(blue, 0.82)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.axis("off")
    T = dict(fontsize=fs, linespacing=1.2)
    v1, v2, arrow = r"$v_1$", r"$v_2$", r"$\rightarrow$"

    # ---- the window: from T1's conflict check to the linking of T1's element
    ax.add_patch(Rectangle((X_CHECK, 0.125), X_LINK - X_CHECK, 0.765, fc=st.lighten(red, 0.87), ec="none", zorder=0))
    for x in (X_CHECK, X_LINK):
        ax.plot([x, x], [0.125, 0.89], color=red, lw=0.5, ls=(0, (2, 1.6)), zorder=1)
    ax.annotate("", xy=(X_CHECK, 0.925), xytext=(X_LINK, 0.925),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.5, color=red, shrinkA=0, shrinkB=0, mutation_scale=4.5))
    ax.text((X_CHECK + X_LINK) / 2, 0.972, "unsoundness window", ha="center", va="center", color=red, fontweight="bold", **T)

    # ---- lanes (a lifeline ends with the transaction)
    for y, name, role, x1 in ((T1Y, "T1", "writer", 91.0), (T2Y, "T2", "reader", 65.8)):
        ax.plot([9.0, x1], [y, y], color="#9A9A9A", lw=0.6, zorder=1)
        ax.text(0.0, y + 0.03, name, ha="left", va="center", fontsize=st.FONT, fontweight="bold", color=ink)
        ax.text(0.0, y - 0.048, role, ha="left", va="center", fontsize=fs, color=mut)

    # ---- T1: its own search, then INSERT = conflict check, graph search, link; COMMIT
    _box(ax, 10.0, 21.5, T1Y, "search", "white", ink, fs)
    ax.text(15.75, T1Y - 0.125, "no duplicate", ha="center", va="center", color=mut, **T)
    _tick(ax, X_CHECK, T1Y, ink)
    ax.text(X_CHECK - 1.0, T1Y + 0.125, "conflict check", ha="right", va="center", color=ink, **T)
    _box(ax, X_CHECK + 1.3, X_LINK - 1.6, T1Y, "INSERT " + v1 + ": graph search", fill, blue, fs)
    ax.plot([X_LINK], [T1Y], marker="o", ms=4.4, mfc=blue, mec=blue, zorder=5)
    ax.text(X_LINK + 1.2, T1Y + 0.125, "link " + v1, ha="left", va="center", color=blue, **T)
    _tick(ax, 91.0, T1Y, ink)
    ax.text(91.0, T1Y + 0.125, "COMMIT", ha="center", va="center", color=ink, **T)

    # ---- where the patch repeats the check
    ax.plot([X_PATCH], [T1Y], marker="D", ms=3.6, mfc="white", mec=green, mew=1.0, zorder=5)
    ax.plot([X_PATCH, X_PATCH + 1.6], [T1Y - 0.04, T1Y - 0.095], color=green, lw=0.45)
    ax.text(X_PATCH + 1.4, T1Y - 0.19, "patch: repeat the\ncheck here; it finds\nT2's lock, T1 aborts", ha="left", va="center", color=green, **T)

    # ---- T2: SIREAD lock on the relation, scan, INSERT, COMMIT, all inside the window
    x_lock = 35.0
    ax.plot([x_lock], [T2Y], marker="s", ms=4.0, mfc="white", mec=ink, mew=0.9, zorder=5)
    ax.plot([x_lock - 0.6, X_CHECK - 0.6], [T2Y - 0.04, T2Y - 0.125], color=mut, lw=0.45)
    ax.text(X_CHECK - 1.2, T2Y - 0.15, "SIREAD lock on\nthe whole index", ha="right", va="center", color=ink, **T)
    _box(ax, x_lock + 1.9, 48.0, T2Y, "scan", "white", ink, fs)
    ax.text(42.7, T2Y - 0.125, "misses " + v1, ha="center", va="center", color=mut, **T)
    _box(ax, 49.2, 64.6, T2Y, "INSERT " + v2, fill, blue, fs)
    _tick(ax, 66.0, T2Y, ink)
    ax.text(66.8, T2Y - 0.125, "COMMIT", ha="right", va="center", color=ink, **T)

    # ---- the dependency that is never recorded: T2 read before T1's write became visible
    ax.add_patch(FancyArrowPatch((43.0, T2Y + 0.068), (X_LINK - 0.9, T1Y - 0.05), connectionstyle="arc3,rad=-0.10", arrowstyle="-|>",
                                 mutation_scale=5.5, lw=0.8, color=red, ls=(0, (3, 1.6)), zorder=4))
    ax.text(48.5, 0.535, "rw-dependency\nT2 " + arrow + " T1 is\nnever recorded", ha="right", va="center", color=red, **T)

    # ---- outcome and time arrow
    ax.text(98.5, T2Y - 0.005, "both commit:\n" + v1 + " and " + v2 + " are\nduplicates", ha="right", va="center", color=ink, fontweight="bold", **T)
    ax.annotate("", xy=(98.5, 0.055), xytext=(9.0, 0.055),
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color=mut, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(98.5, 0.095, "time", ha="right", va="center", color=mut, **T)


def main(results="results", suffix="", out="figures"):
    fig, ax = st.subplots(cols=1, aspect=0.60)
    draw(ax)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
