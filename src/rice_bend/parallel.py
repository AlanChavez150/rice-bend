"""One parallel map, replacing four hand-rolled ProcessPoolExecutor blocks.

Each of those four came paired with a serial arm that re-implemented the worker body
by hand: ~170 lines and 8 code paths expressing one idea, with the two arms free to
drift (and they had — one swallowed the exception message the other printed). Here
`jobs=1` runs the *same* task function the pool would, so they cannot.

Results are assembled by input index, so `map_workers` returns input order whatever
the completion order was, while `on_done` still fires as each task lands — the live
[done/total] feedback matters when the sweep is the multi-hour part.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence

# Read-only payload handed to every task, set once per worker by the initializer so
# it is not re-pickled for each of the (potentially thousands of) tasks.
_SHARED = None


def worker_shared():
    """The `shared` payload map_workers was given. Call from inside a task."""
    return _SHARED


def _init_worker(shared, setup: Optional[Callable[[], None]]) -> None:
    """Pool initializer. Runs in the worker process ONLY — never in the parent, so
    a serial run cannot inherit a side effect (e.g. switching matplotlib to Agg,
    which would permanently kill --show for the rest of the process)."""
    global _SHARED
    # These do NOT limit the BLAS numpy has already loaded, and that is measured,
    # not assumed: the parent imported numpy before the pool forked, OpenBLAS sized
    # its pool to every core at that import, and a variable set afterwards does not
    # shrink it. With these lines in place a matmul on the per-candidate path
    # (mgs._offset_peaks' score, as first written) ran on 32 threads in every
    # worker: a three-tone 441-candidate patch at 4 workers (rust engine) took
    # 9 min 01 s of user and 10 min 57 s of system time, against 5 min 39 s and
    # 9 s once the matmul was gone (the 9 s is the parent's own analysis) — the
    # same as exporting OPENBLAS_NUM_THREADS=1 before python starts, which is the
    # only place the variable binds.
    #
    # What keeps each worker on one thread is therefore the task, not this loop:
    # the per-candidate path (gs_reconstruct and its starts) is fftconvolve +
    # ufuncs + explicit sums, with no `@`/dot/vdot/einsum and no linalg/polyfit on
    # it (see mgs._offset_peaks, mgs._line_fit); the scene-
    # rendering tasks make no such call of their own either. A BLAS or LAPACK call
    # added to a task brings the oversubscription back at once and silently — the
    # results do not change, only the system time. BLAS that runs in the PARENT is
    # fine and exists (analysis.py's SVD, landing.py's norms).
    #
    # The variables are still set, for whatever reads them late: numexpr, a
    # library first imported inside a worker, a process a worker starts. They are
    # deliberately NOT set in __init__.py — the one placement where they would bind
    # numpy's BLAS, which is exactly the problem: __init__.py runs for plain `mgs`
    # too, so a serial solve would newly be pinned to one math thread as an
    # import-time side effect.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if setup is not None:
        setup()
    _SHARED = shared


def map_workers(task: Callable, items: Sequence, *, jobs: int = 1, shared=None,
                setup: Optional[Callable[[], None]] = None,
                on_done: Optional[Callable] = None,
                log: Optional[logging.Logger] = None,
                desc: Optional[str] = None) -> List:
    """Run `task(item)` for every item, returning the results in INPUT order.

    `jobs <= 1` (or a single item) runs in this process; otherwise the work is spread
    over `min(jobs, len(items))` processes. `task` must be a module-level function —
    it is pickled — and reads `shared` via `worker_shared()`.

    `on_done(done, total, result)` fires as each result lands (`done` counts
    completions, not indices). `setup` runs once per worker process, in the pool only.
    """
    items = list(items)
    total = len(items)
    results: List = [None] * total
    if total == 0:
        return results
    n_jobs = max(1, int(jobs))

    if n_jobs == 1 or total <= 1:
        if log is not None and desc:
            log.info(desc)
        global _SHARED
        prev, _SHARED = _SHARED, shared
        try:
            for i, item in enumerate(items):
                results[i] = task(item)
                if on_done is not None:
                    on_done(i + 1, total, results[i])
        finally:
            _SHARED = prev
        return results

    n_workers = min(n_jobs, total)
    if log is not None and desc:
        log.info(f"{desc} across {n_workers} worker process(es)")
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(shared, setup)) as ex:
        futures = {ex.submit(task, item): i for i, item in enumerate(items)}
        for done, fut in enumerate(as_completed(futures), start=1):
            i = futures[fut]
            results[i] = fut.result()
            if on_done is not None:
                on_done(done, total, results[i])
    return results


def round_robin_chunks(items: Sequence, n_chunks: int) -> List[List]:
    """Stride `items` across `n_chunks` chunks. Every task here costs the same, so
    striding balances the load; one chunk per worker keeps a chunk's returned
    partial sum small."""
    n_chunks = max(1, min(int(n_chunks), len(items)))
    return [list(items[i::n_chunks]) for i in range(n_chunks)]
