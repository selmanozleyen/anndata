"""
This module will benchmark io of AnnData objects

Things to test:

* Read time, write time
* Peak memory during io
* File sizes

Parameterized by:

* What method is being used
* What data is being included
* Size of data being used

Also interesting:

* io for views
* io for backed objects
* Reading dense as sparse, writing sparse as dense
"""
from pathlib import Path

import numpy as np



import anndata

import dask.array as da
import zarr

from anndata.experimental import read_dispatched, read_elem




TOY_DATA = [
    Path(__file__).parent / "data/03_fat_np.zarr",
    # Path(__file__).parent / "data/06_tall_np.zarr",
    # Path(__file__).parent / "data/09_square_np.zarr",
    # Path(__file__).parent / "data/18_square_np.zarr",
    # Path(__file__).parent / "data/27_square_np.zarr",
]


class NoSetup:
    def setup(self, *args, **kwargs):
        raise NotImplementedError("Must implement setup")


class FuncSuiteBase(NoSetup):
    def time_func_full(self, filepath):
        self._load()
        self._run()

    def peakmem_func_full(self, filepath):
        self._load()
        self._run()


def read_dask(store):
    f = zarr.open(store, mode="r")

    def callback(func, elem_name: str, elem, iospec):
        if iospec.encoding_type in (
            "dataframe",
            "csr_matrix",
            "csc_matrix",
            "awkward-array",
        ):
            # Preventing recursing inside of these types
            return read_elem(elem)
        elif iospec.encoding_type == "array":
            return da.from_zarr(elem)
        else:
            return func(elem)

    adata = read_dispatched(f, callback=callback)

    return adata


class SetupBase:
    params = [*TOY_DATA]
    param_names = ["filepath"]

    def setup(self, filepath):
        self.filepath = filepath


class NumpyLoadBase(NoSetup):
    def _load(self):
        self.adata = anndata.read_zarr(self.filepath)
        return self.adata


class DaskLoadBase(NoSetup):
    def _load(self):
        self.adata = read_dask(self.filepath)
        return self.adata


class DaskSVDFunc(NoSetup):
    def _run(self):
        u, s, v = da.linalg.svd(self.adata.X)
        u.compute(), s.compute(), v.compute()


class DaskSumFunc(NoSetup):
    def _run(self):
        res = self.adata.X.sum()
        res.compute()


class NumpySVDFunc(NoSetup):
    def _run(self):
        u, s, v = np.linalg.svd(self.adata.X)


class NumpySumFunc(NoSetup):
    def _run(self):
        res = np.sum(self.adata.X)


class DaskSVDSuite(SetupBase, DaskLoadBase, DaskSVDFunc, FuncSuiteBase):
    timeout=240


class NumpySVDSuite(SetupBase, NumpyLoadBase, NumpySVDFunc, FuncSuiteBase):
    timeout=240
