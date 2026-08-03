"""Force a headless matplotlib backend for batch planner runs.

polyfly_ral's global_planner.py hardcodes matplotlib.use('TkAgg') at import
time, which fails without an X display. We neutralize interactive backend
selection instead of patching the upstream repo.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    _INTERACTIVE = {"TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg", "MacOSX", "WXAgg"}

    def _use(backend, *a, **k):
        if str(backend) in _INTERACTIVE:
            return None
        return _orig_use(backend, *a, **k)

    _orig_use = matplotlib.use
    matplotlib.use = _use
except Exception:
    pass
