from pathlib import Path

import numpy as np
import tempfile

import glob, os
import anndata
from anndata._io.merge import concat_on_disk
import dask.array as da
import zarr

from anndata.experimental import read_dispatched, read_elem

NPS = set(glob.glob(str(Path(__file__).parent) + "/data/*np*"))
CSRS = set(glob.glob(str(Path(__file__).parent) + "/data/*csr*"))
CSCS = set(glob.glob(str(Path(__file__).parent) + "/data/*csc*"))
FATS = set(glob.glob(str(Path(__file__).parent) + "/data/*fat*"))
TALLS = set(glob.glob(str(Path(__file__).parent) + "/data/*tall*"))
SQUARES = set(glob.glob(str(Path(__file__).parent) + "/data/*square*"))


class NoSetup:
    def setup(self, *args, **kwargs):
        raise NotImplementedError("Must implement setup")


class WriteSuiteBase(NoSetup):
    def _setup(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.writepth = Path(self.tmpdir.name) / "out.zarr"

    def _teardown(self, *args, **kwargs):
        self.tmpdir.cleanup()


class FuncSuiteBase(NoSetup):
    def time_func_full(self, *args, **kwargs):
        concat_on_disk(self.filepaths, self.writepth, axis=self.axis)

    def peakmem_func_full(self, *args, **kwargs):
        concat_on_disk(self.filepaths, self.writepth, axis=self.axis)


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


class ByArrayTypeSuite(WriteSuiteBase, FuncSuiteBase):
    params = [
        ("csrs", "nps-0", "nps-1", "cscs"),
    ]
    param_names = ["fileset"]

    def setup(self, fileset):
        if "csrs" in fileset:
            self.filepaths = CSRS
            self.axis = 0
        elif "nps" in fileset:
            self.filepaths = NPS
            if "0" in fileset:
                self.axis = 0
            elif "1" in fileset:
                self.axis = 1
        elif fileset == "cscs":
            self.filepaths = CSCS
            self.axis = 1

        if self.axis == 0:
            self.filepaths = self.filepaths.intersection(FATS.union(SQUARES))
        elif self.axis == 1:
            self.filepaths = self.filepaths.intersection(TALLS.union(SQUARES))

        self._setup()

    def teardown(self, *args, **kwargs):
        self._teardown()
