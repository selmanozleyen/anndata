"""A zarr array is read with its integer axis sorted, and the caller's order restored.

The h5py path has always sorted, because HDF5 refuses an unsorted selection. zarr accepts
one, so nothing ever forced the issue -- and a codec pipeline that can only describe a
non-decreasing axis (zarrs) quietly declines the read and lets zarr-python serve it.

Correctness alone does not test this: the unsorted read returned the right rows before the
change too. What has to be asserted is that the STORE saw a non-decreasing selection.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from anndata._core.index import _subset


@pytest.fixture
def arr() -> zarr.Array:
    z = zarr.create_array(
        store=zarr.storage.MemoryStore(), shape=(64, 8), chunks=(8, 8), dtype="int32"
    )
    z[:] = np.arange(64 * 8, dtype="int32").reshape(64, 8)
    return z


@pytest.fixture
def seen(monkeypatch) -> list:
    """Every selection the store is actually asked for."""
    calls: list = []
    original = zarr.Array.__getitem__

    def spy(self, selection):
        calls.append(selection)
        return original(self, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", spy)
    return calls


def _axis0(selection) -> np.ndarray | None:
    sel = selection if isinstance(selection, tuple) else (selection,)
    return sel[0] if isinstance(sel[0], np.ndarray) else None


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(np.array([9, 2, 40, 17]), id="unsorted"),
        pytest.param(np.array([40, 17, 9, 2]), id="descending"),
        # Duplicates stay a permutation under a stable sort, and a repeated index leaves
        # the axis non-decreasing, which is all the pipeline asks of it.
        pytest.param(np.array([9, 2, 9, 40, 2]), id="duplicates"),
        pytest.param(np.array([2, 9, 17, 40]), id="already-sorted"),
        pytest.param(np.array([7]), id="single"),
    ],
)
def test_rows_come_back_in_the_order_asked_for(arr, seen, rows) -> None:
    got = _subset(arr, (rows, slice(None)))
    np.testing.assert_array_equal(got, np.asarray(arr[:])[rows])
    # The point of the change: whatever order the caller used, the store saw one it can
    # describe. Without the sort this fails on every case but the last two.
    axis0 = [a for a in map(_axis0, seen) if a is not None]
    assert axis0, f"no array selection reached the store: {seen}"
    for a in axis0:
        assert bool((a[1:] >= a[:-1]).all()), f"store saw a non-monotonic axis: {a}"


def test_a_column_axis_is_sorted_too(arr, seen) -> None:
    """The reordered axis is whichever one is an array, not axis 0 by assumption."""
    cols = np.array([6, 1, 4])
    got = _subset(arr, (slice(None), cols))
    np.testing.assert_array_equal(got, np.asarray(arr[:])[:, cols])
    checked = 0
    for sel in seen:
        # `seen` also holds the plain `arr[:]` the assertion above makes, which is not a
        # tuple and has no axis 1.
        if not isinstance(sel, tuple) or len(sel) < 2:
            continue
        if isinstance(sel[1], np.ndarray):
            assert bool((sel[1][1:] >= sel[1][:-1]).all()), sel[1]
            checked += 1
    assert checked, f"no array selection on axis 1 reached the store: {seen}"


def test_two_array_axes_are_left_alone(arr, seen) -> None:
    """Two array axes index coordinate-wise through np.ix_, where a per-axis permutation
    is not a permutation of the result. Correctness is the whole contract here."""
    rows, cols = np.array([9, 2, 40]), np.array([6, 1, 4])
    got = _subset(arr, (rows, cols))
    np.testing.assert_array_equal(got, np.asarray(arr[:])[np.ix_(rows, cols)])


def test_a_boolean_mask_is_left_alone(arr, seen) -> None:
    mask = np.zeros(64, dtype=bool)
    mask[[2, 9, 40]] = True
    got = _subset(arr, (mask, slice(None)))
    np.testing.assert_array_equal(got, np.asarray(arr[:])[mask])
