"""Anatomical masks: exact base-region matching (reviewer 2, point 6)."""

from __future__ import annotations

import numpy as np
import pytest

from hoc.atlas import build_label_table, classify_region_group, region_set_mask
from hoc.config import load_config
from hoc.paths import REPO_ROOT
from hoc.targets import build_targets


@pytest.fixture(scope="module")
def anatomy() -> dict:
    return load_config(REPO_ROOT / "config" / "default.toml")["anatomy"]


@pytest.mark.parametrize("scale", [68, 114, 219])
def test_precuneus_is_association_only(anatomy, scale) -> None:
    labels = build_label_table(scale, "synthetic:")
    occipital = region_set_mask(labels, anatomy["occipital_regions"])
    association = region_set_mask(labels, anatomy["association_regions"])
    base = labels["base_region"]
    assert not occipital[base.eq("precuneus").to_numpy()].any()
    assert association[base.eq("precuneus").to_numpy()].all()
    assert occipital[base.eq("cuneus").to_numpy()].all()
    assert not np.any(occipital & association)
    assert set(base[occipital]) == {"cuneus", "lateraloccipital", "lingual", "pericalcarine"}


def test_anatomical_group_of_precuneus() -> None:
    assert classify_region_group("precuneus") == "parietal"
    assert classify_region_group("cuneus") == "occipital"


def test_unknown_region_is_an_error() -> None:
    labels = build_label_table(68, "synthetic:")
    with pytest.raises(ValueError, match="not in the atlas"):
        region_set_mask(labels, ["cuneus", "precuneous"])


def test_module_targets_are_disjoint_unit_vectors(anatomy) -> None:
    labels = build_label_table(68, "synthetic:")
    targets = build_targets(np.eye(68), labels, {}, anatomy, "synthetic:")
    occipital, association = targets["occipital_module"], targets["association_module"]
    assert np.linalg.norm(occipital) == pytest.approx(1.0)
    assert np.dot(occipital, association) == 0.0
    contrast = targets["association_vs_occipital"]
    assert np.all(contrast[occipital > 0] < 0) and np.all(contrast[association > 0] > 0)
    # sign convention: the gradient points from occipital (low) to association (high)
    gradient = targets["principal_gradient"]
    assert gradient[association > 0].mean() > gradient[occipital > 0].mean()
