"""Small analysis tools that work on run logs.

  python -m ariadne.tools.redraw RUN.pkl 1 2 3 ...            keep the timing of a run (commit log, read and validation positions),
                                                             redraw the workload with other seeds, redo the extraction of
                                                             background in-ball insertions (rate and rank histogram).  Calibrates
                                                             how much the rank-1 share of background insertions scatters from
                                                             seed to seed.  For runs in which every task inserts (tau_cos: 0).
  python -m ariadne.tools.redraw_threshold RUN.pkl 1 2 ...   the same for runs with the threshold decider: every decision is
                                                             re-taken on the redrawn geometry.
"""
