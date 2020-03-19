import pytest
import numpy as np
import pandas as pd
import scipy.sparse as ssp

from anndata import AnnData
from anndata._core.ref_path import RefPath


@pytest.fixture
def adata():
    return AnnData(
        X=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        layers=dict(unspliced=np.array([[0.1, 1.2, 2.3, 3.4], [4.5, 5.6, 6.7, 7.8]])),
        obs=dict(obs_names=["Cell1", "Cell2"], group=["batch1", "batch2"]),
        obsm=dict(
            X_pca=np.array([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]]),
            protein=pd.DataFrame(dict(CD14=[2.1, 3.2])),
        ),
        obsp=dict(neighbors_distances=ssp.csr_matrix([[0, 0.4], [0.3, 0]])),
        var=dict(var_names=[f"Gene{i}" for i in "WXYZ"], mito=[0.4, 0.3, 0.2, 0.1]),
        # varm=dict(...),
        varp=dict(cors=np.array([(5, 4, 3, 2)] * 4)),
        raw=dict(
            X=np.array([[0, 11, 22, 33, 88], [44, 55, 66, 77, 99]]),
            var=dict(
                var_names=[f"Gene{i}" for i in "WXYZA"],
                symbol=[f"Symb{i}" for i in range(1, 6)],
            ),
        ),
    )


paths = [
    (("layers", "unspliced", "obs", "Cell1"), [0.1, 1.2, 2.3, 3.4]),
    (("obs", "group"), ["batch1", "batch2"]),
    (("obsm", "X_pca", 1), [0.4, 0.3]),
    (("obsm", "protein", "CD14"), [2.1, 3.2]),
    (("obsp", "neighbors_distances", "Cell2", 0), [0.3, 0]),
    (("var", "mito"), [0.4, 0.3, 0.2, 0.1]),
    # (("varm", "", ""), []),
    (("varp", "cors", "GeneY", 1), [3] * 4),
    (("raw", "X", "var", "GeneA"), [88, 99]),
    (("raw", "var", "symbol"), ["Symb1", "Symb2", "Symb3", "Symb4", "Symb5"]),
]


@pytest.mark.parametrize("args,expected", paths, ids=repr)
def test_instantiate(args, expected):
    RefPath(*args)


@pytest.mark.parametrize("args,expected", paths, ids=repr)
def test_get_vector(args, expected, adata):
    rp = RefPath(*args)
    vec = rp.get_vector(adata)
    assert isinstance(vec, np.ndarray)
    assert len(vec.shape) == 1
    assert vec.tolist() == expected


def test_alias():
    raise NotImplementedError()


@pytest.mark.parametrize(
    "rp_code", ["RefPath('obs', 'group')", "RefPath('varp', 'neighbors', 'Cell5', 1)"]
)
def test_repr(rp_code):
    rp = eval(rp_code)
    assert repr(rp) == rp_code


@pytest.mark.parametrize(
    "spec", [(("X", "var", "GeneX"), ("layers", "X", "var", "GeneX")),]
)
def test_parse(spec):
    raise NotImplementedError()
