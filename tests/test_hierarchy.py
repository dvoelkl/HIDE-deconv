"""
=====================================================
Tests for the cell type hierarchy
=====================================================
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from hide_deconv.preprocessing import (
    create_hierarchy,
    fill_missing_celltypes,
    validate_hierarchy,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def make_adata(obs: dict) -> ad.AnnData:
    n_cells = len(next(iter(obs.values())))
    return ad.AnnData(
        X=np.ones((n_cells, 2)),
        obs=pd.DataFrame(obs, index=[f"cell_{i}" for i in range(n_cells)]),
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_valid_hierarchy_multiple_children_per_parent():
    adata = make_adata(
        {
            "sub": ["a1", "a2", "b1", "b1"],
            "major": ["a", "a", "b", "b"],
        }
    )

    A = create_hierarchy(adata, "sub", ["major"])["major"]

    assert A.loc["a", ["a1", "a2"]].tolist() == [1, 1]
    assert A.loc["b", "b1"] == 1
    assert A.sum(axis=0).tolist() == [1, 1, 1]


def test_child_with_multiple_parents_raises():
    adata = make_adata(
        {
            "sub": ["a1", "a1", "a1", "b1"],
            "major": ["a", "a", "b", "b"],
        }
    )

    with pytest.raises(ValueError) as e:
        create_hierarchy(adata, "sub", ["major"])

    msg = str(e.value)
    assert "'sub' and 'major'" in msg
    assert "'a1' -> 'a' (2 cells), 'b' (1 cells)" in msg


def test_validate_checks_neighboring_levels():
    # sub -> mid is unique, but mid "m1" belongs to two coarse cell types
    adata = make_adata(
        {
            "sub": ["s1", "s2"],
            "mid": ["m1", "m1"],
            "coarse": ["c1", "c2"],
        }
    )

    with pytest.raises(ValueError, match="'mid' and 'coarse'"):
        validate_hierarchy(adata, ["sub", "mid", "coarse"])


def test_missing_child_inherits_parent():
    adata = make_adata(
        {
            "sub": ["a1", np.nan, "", np.nan],
            "mid": ["a", "a", np.nan, np.nan],
            "coarse": ["A", "A", "A", "B"],
        }
    )

    with pytest.warns(UserWarning):
        fill_missing_celltypes(adata, ["sub", "mid", "coarse"])

    assert adata.obs["mid"].astype(str).tolist() == ["a", "a", "A", "B"]
    assert adata.obs["sub"].astype(str).tolist() == ["a1", "a", "A", "B"]

    validate_hierarchy(adata, ["sub", "mid", "coarse"])


def test_fill_without_missing_labels_keeps_obs():
    adata = make_adata({"sub": ["a1", "b1"], "major": ["a", "b"]})
    before = adata.obs.copy()

    fill_missing_celltypes(adata, ["sub", "major"])

    pd.testing.assert_frame_equal(adata.obs, before)
