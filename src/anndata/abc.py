from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import ClassVar, Literal

    import numpy as np

    from ._types import _GroupStorageType
    from .compat import CSArray, CSMatrix
    from .typing import Index


__all__ = ["CSCDataset", "CSRDataset"]


class _AbstractCSDataset(ABC):
    """Base for the public API for CSRDataset/CSCDataset."""

    format: ClassVar[Literal["csr", "csc"]]
    """The format of the sparse matrix."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Shape of the matrix."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The :class:`numpy.dtype` of the `data` attribute of the sparse matrix."""

    @property
    @abstractmethod
    def indices_dtype(self) -> np.dtype:
        """The :class:`numpy.dtype` of the `indices` attribute of the sparse matrix.

        With :attr:`dtype` and :attr:`indptr`, this is everything needed to size a read
        before making it.
        """

    @property
    @abstractmethod
    def backend(self) -> Literal["zarr", "hdf5"]:
        """Which file type is used on-disk."""

    @property
    @abstractmethod
    def group(self) -> _GroupStorageType:
        """The group underlying the backed matrix."""

    @property
    @abstractmethod
    def indptr(self) -> np.ndarray:
        """Boundaries of the vectors along the major axis.

        Public because sizing a read needs it before the read happens: a caller
        preallocating one buffer across several datasets has to know each one's nnz up
        front. Normally cached on first access, so free thereafter.
        """

    @abstractmethod
    def __getitem__(self, index: Index) -> float | CSMatrix | CSArray:
        """Load a slice or an element from the sparse dataset into memory.

        Parameters
        ----------
        index
            Index to load.

        Returns
        -------
        The desired data read off disk.
        """

    @abstractmethod
    def to_memory(self) -> CSMatrix | CSArray:
        """Load the sparse dataset into memory.

        Returns
        -------
        The in-memory representation of the sparse dataset.
        """

    @abstractmethod
    def append(
        self, sparse_matrix: CSMatrix | CSArray | CSRDataset | CSCDataset
    ) -> None:
        """Append an in-memory or on-disk sparse matrix to the current object's store.

        Parameters
        ----------
        sparse_matrix
            The matrix to append.

        Raises
        ------
        NotImplementedError
            If the matrix to append is not one of :class:`~scipy.sparse.csr_array`, :class:`~scipy.sparse.csc_array`, :class:`~scipy.sparse.csr_matrix`, or :class:`~scipy.sparse.csc_matrix`.
        ValueError
            If both the on-disk and to-append matrices are not of the same format i.e., `csr` or `csc`.
        OverflowError
            If the underlying data store has a 32 bit indptr, and the new matrix is too large to fit in it i.e., would cause a 64 bit `indptr` to be written.
        AssertionError
            If the on-disk data does not have `csc` or `csr` format.
        """


_sparse_dataset_doc = """\
On disk {format} sparse matrix.

Analogous to :class:`h5py.Dataset` or :class:`zarr.Array`, but for sparse matrices.
"""


def _redeclare_abstract_methods[T: type[_AbstractCSDataset]](cls: T) -> T:
    """Rebind the abstract methods so Sphinx doesn’t interpret them as inherited."""
    for name in ("__getitem__", "to_memory", "append", "indptr", "indices_dtype"):
        setattr(cls, name, getattr(_AbstractCSDataset, name))
    return cls


@_redeclare_abstract_methods
class CSRDataset(_AbstractCSDataset, ABC):
    __doc__ = _sparse_dataset_doc.format(format="CSR")
    format = "csr"

    @abstractmethod
    async def aread_rows(
        self, rows: np.ndarray, *, out: tuple[np.ndarray, np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read whole rows, concurrently, optionally into buffers the caller owns.

        The same read as ``self[rows]``, but awaitable. zarr runs a single shared event
        loop, so the synchronous form deadlocks if called from inside a coroutine
        already running on it: a caller gathering reads across several datasets cannot
        use `__getitem__` at all, and can await this one alongside the rest of its work.

        Rows are read once each and ascending, whatever order was asked for, because a
        backed read is only described efficiently as ordered contiguous ranges. The
        result is still in the caller's order, repeats included.

        Parameters
        ----------
        rows
            Row indices, in any order, repeats allowed.
        out
            ``(data, indices)`` buffers to read into, each as long as the total nnz of
            `rows` -- sized by the caller from :attr:`indptr`. The read lands in them
            directly when `rows` is already ascending and distinct, and is placed into
            them otherwise.

        Returns
        -------
        The `data`, `indices` and `indptr` of the selected rows.

        Raises
        ------
        TypeError
            If the store has no asynchronous form, as for HDF5.
        """


@_redeclare_abstract_methods
class CSCDataset(_AbstractCSDataset, ABC):
    __doc__ = _sparse_dataset_doc.format(format="CSC")
    format = "csc"
