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
    # Precautionary and currently inert: the GS hot path is fftconvolve + ufuncs,
    # with zero matmul/dot/linalg/einsum anywhere in src/, so there is no BLAS to
    # oversubscribe. Kept here, and deliberately NOT in __init__.py — that is the
    # only placement where these would take effect, which is exactly the problem:
    # __init__.py runs for plain `mgs` too, so a serial solve would newly be pinned
    # to one math thread as an import-time side effect.
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
