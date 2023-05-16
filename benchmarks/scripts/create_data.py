import numpy as np
from scipy import sparse
import pandas as pd
from anndata.tests.helpers import gen_typed_df
from anndata.experimental import read_dispatched, write_dispatched, read_elem
import zarr
import anndata

shapes = ["fat", "tall", "square"]
sizes = [10_000]
densities = [0.1, 1]
NUM_RUNS = 3


def create_adata(shape, X):
    M, N = shape
    obs_names = pd.Index(f"cell{i}" for i in range(shape[0]))
    var_names = pd.Index(f"gene{i}" for i in range(shape[1]))
    obs = gen_typed_df(M, obs_names)
    var = gen_typed_df(N, var_names)
    # For #147
    obs.rename(columns=dict(cat="obs_cat"), inplace=True)
    var.rename(columns=dict(cat="var_cat"), inplace=True)
    return anndata.AnnData(X, obs=obs, var=var)


def write_chunked(func, store, k, elem, dataset_kwargs, iospec):
    """Write callback that chunks X and layers"""

    def set_chunks(d, chunks=None):
        """Helper function for setting dataset_kwargs. Makes a copy of d."""
        d = dict(d)
        if chunks is not None:
            d["chunks"] = chunks
        else:
            d.pop("chunks", None)
        return d

    if iospec.encoding_type == "array":
        if 'layers' in k or k.endswith('X'):
            dataset_kwargs = set_chunks(dataset_kwargs, (adata.shape[0], 25))
        else:
            dataset_kwargs = set_chunks(dataset_kwargs, None)

    func(store, k, elem, dataset_kwargs=dataset_kwargs)



if __name__ == "__main__":
    file_id = 1
    for _ in range(NUM_RUNS):
        for shape in shapes:
            for size in sizes:
                for density in densities:
                    is_dense = density == 1
                    array_funcs = []
                    array_names = []
                    if is_dense:
                        array_names.append("np")
                        array_funcs.append(lambda x: x.toarray())
                    else:
                        array_names.append("csc")
                        array_names.append("csr")
                        array_funcs.append(sparse.csc_matrix)
                        array_funcs.append(sparse.csr_matrix)

                    for array_func, array_name in zip(array_funcs, array_names):
                        M = size
                        N = size
                        if shape != "square":
                            other_size = int(size * np.random.uniform(0.7, 0.9))
                            if shape == "fat":
                                M = other_size
                            elif shape == "tall":
                                N = other_size

                        X = array_func(
                            sparse.random(M, N, density=density, format="csc")
                        )
                        adata = create_adata(
                            (M, N),
                            X,
                        )
                        fname = f"benchmarks/data/{file_id:02d}_{shape}_{array_name}"
                        file_id += 1
                        print(f"{M}x{N}_density={density:0.1f}_{array_name} -> {fname}")
                        # adata.write_h5ad(f"{fname}.h5ad")
                        if is_dense:
                            output_zarr_path = f"{fname}.zarr"
                            z = zarr.open_group(output_zarr_path)

                            write_dispatched(z, "/", adata, callback=write_chunked)
                            zarr.consolidate_metadata(z.store)
                        else:
                            adata.write_zarr(f"{fname}.zarr")
