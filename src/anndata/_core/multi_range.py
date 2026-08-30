"""Reading several contiguous ranges of one array as a single zarr selection.

zarr describes a read as ONE selection, so N disjoint ranges would otherwise mean N
calls, and the per-call overhead grows with the range count -- which is precisely the
regime a backed row read lives in. `_MultiRangeIndexer` presents the ranges as a single
selection and re-bases each chunk projection onto one output buffer, so they land in it
back to back, in the order given.

This lives in its own module because it is NOT a sparse concern: the same construct
serves a dense row read (runs of whole rows on the leading axis) and a CSR one (runs of
`data`/`indices` derived from `indptr`). It existed twice -- here and, in near-identical
form, in annbatch -- which is two copies of a workaround for the same missing zarr
feature. One copy can at least be fixed once.

None of this would be needed if zarr exposed a public multi-range selection; until it
does, `zarr.core.indexing.Indexer` is subclassed here and nowhere else in anndata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import zarr
import zarr.core.buffer
import zarr.core.indexing

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterator, Sequence
    from typing import Any

    from zarr.core.buffer import BufferPrototype, NDBuffer
    from zarr.core.common import NDArrayLikeOrScalar


class _MultiRangeIndexer(zarr.core.indexing.Indexer):
    """Several contiguous ranges of an array's LEADING axis, read as ONE selection.

    The trailing axes are taken whole, so on a 1-D array this is a set of element ranges
    and on a 2-D one it is a set of whole-row bands. Both are the same construct, which
    is why the dense and sparse readers do not need one each.
    """

    def __init__(self, arr: zarr.Array, runs: Sequence[slice]) -> None:
        # `Array._chunk_grid` since zarr 3.1.7; on the metadata before that.
        chunk_grid = getattr(arr, "_chunk_grid", None) or arr.metadata.chunk_grid
        # `Ellipsis` is a no-op on a 1-D array and takes the trailing axes whole on any
        # other, so one indexer covers every rank without a special case per rank.
        self.indexers = [
            zarr.core.indexing.BasicIndexer(
                (run, Ellipsis), shape=arr.metadata.shape, chunk_grid=chunk_grid
            )
            for run in runs
        ]
        self.shape = (
            sum(i.shape[0] for i in self.indexers),
            *self.indexers[0].shape[1:],
        )
        self.drop_axes = self.indexers[0].drop_axes

    def __iter__(self) -> Iterator[zarr.core.indexing.ChunkProjection]:
        at = 0
        for indexer in self.indexers:
            for proj in indexer:
                width = proj.out_selection[0].stop - proj.out_selection[0].start
                yield type(proj)(
                    proj.chunk_coords,
                    proj.chunk_selection,
                    (slice(at, at + width), *proj.out_selection[1:]),
                    proj.is_complete_chunk,
                )
                at += width


def aread_ranges(
    arr: zarr.Array,
    runs: Sequence[slice],
    *,
    prototype: BufferPrototype | None = None,
    out: NDBuffer | None = None,
) -> Coroutine[Any, Any, NDArrayLikeOrScalar]:
    """Read several contiguous ranges of `arr` as ONE selection, as a coroutine.

    A call per range would pay a round-trip onto zarr's event loop each time, and ranges
    are exactly what there are many of here.
    """
    if prototype is None:
        prototype = zarr.core.buffer.default_buffer_prototype()
    return arr._async_array._get_selection(
        _MultiRangeIndexer(arr, runs), prototype=prototype, out=out
    )


def read_ranges(
    arr: zarr.Array, runs: Sequence[slice], *, out: NDBuffer | None = None
) -> NDArrayLikeOrScalar:
    """:func:`aread_ranges` for callers that are not themselves on the event loop.

    Not safe from a thread already running zarr's event loop -- the sync bridge would
    deadlock there. A pool worker is not that thread.
    """
    from zarr.core.sync import sync as zarr_sync

    prototype = zarr.core.buffer.default_buffer_prototype()
    return zarr_sync(aread_ranges(arr, runs, prototype=prototype, out=out))
