from __future__ import annotations

import numpy as np

from hoc.control import assemble_gramian, gramian_metrics, single_node_gramians
from hoc.preprocessing import preprocess_structural_adjacency, stabilize_adjacency
from hoc.rankings import top_k_excluding_nodes
from hoc import targets as targets_module
from hoc.targets import build_target_metadata, build_targets, normalize_vector


def test_preprocess_structural_adjacency_is_symmetric_and_normalized() -> None:
    raw = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [0.0, 4.0, 0.0],
        ]
    )
    processed = preprocess_structural_adjacency(raw)
    assert np.allclose(processed, processed.T)
    assert np.allclose(np.diag(processed), 0.0)
    assert processed.max() == 1.0


def test_single_node_gramians_add_for_multi_node_driver_sets() -> None:
    adjacency = np.array(
        [
            [0.0, 0.4, 0.1],
            [0.4, 0.0, 0.2],
            [0.1, 0.2, 0.0],
        ]
    )
    system_matrix, _ = stabilize_adjacency(adjacency, c=0.1)
    node_gramians = single_node_gramians(system_matrix)
    combined = assemble_gramian(node_gramians, np.array([0, 2]))
    expected = node_gramians[0] + node_gramians[2]
    assert np.allclose(combined, expected)


def test_gramian_metrics_are_finite_after_regularization() -> None:
    adjacency = np.array(
        [
            [0.0, 0.3, 0.0],
            [0.3, 0.0, 0.25],
            [0.0, 0.25, 0.0],
        ]
    )
    system_matrix, _ = stabilize_adjacency(adjacency, c=0.1)
    node_gramians = single_node_gramians(system_matrix)
    metrics = gramian_metrics(assemble_gramian(node_gramians, np.array([1])), epsilon=1.0e-5)
    assert np.isfinite(metrics["log10_average_energy"])
    assert np.isfinite(metrics["effective_rank"])
    assert np.isfinite(metrics["log10_condition_number"])


def test_gramian_metrics_report_floor_fraction() -> None:
    gramian = np.diag([0.0, 1.0, 2.0])
    metrics = gramian_metrics(gramian, epsilon=1.0e-5)
    assert metrics["eigenvalues_at_floor"] == 1
    assert np.isclose(metrics["fraction_eigenvalues_at_floor"], 1.0 / 3.0)

    no_floor_metrics = gramian_metrics(gramian, epsilon=0.0, apply_floor=False)
    assert no_floor_metrics["eigenvalues_at_floor"] == 0
    assert no_floor_metrics["fraction_eigenvalues_at_floor"] == 0.0
    assert np.isinf(no_floor_metrics["condition_number"])


def test_normalize_vector_returns_unit_norm() -> None:
    vector = np.array([0.0, 3.0, 4.0])
    normalized = normalize_vector(vector)
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_target_catalog_aggregates_bilateral_landmark_subparcels(monkeypatch) -> None:
    # The principal-gradient target needs the fetched Margulies map and the Cammoun
    # annot files; this test is about landmark aggregation, so stub it out.
    monkeypatch.setattr(
        targets_module,
        "margulies_principal_gradient",
        lambda labels, atlas_dir: np.arange(len(labels), dtype=float),
    )

    labels = np.array(
        [
            (0, 219, "L", "pericalcarine_1", "L_pericalcarine_1", "pericalcarine", "occipital"),
            (1, 219, "R", "pericalcarine_1", "R_pericalcarine_1", "pericalcarine", "occipital"),
            (2, 219, "R", "pericalcarine_2", "R_pericalcarine_2", "pericalcarine", "occipital"),
            (3, 219, "L", "precentral_1", "L_precentral_1", "precentral", "frontal"),
            (4, 219, "R", "precentral_1", "R_precentral_1", "precentral", "frontal"),
            (5, 219, "L", "precuneus_1", "L_precuneus_1", "precuneus", "parietal"),
            (6, 219, "R", "posteriorcingulate_1", "R_posteriorcingulate_1", "posteriorcingulate", "cingulate"),
            (7, 219, "L", "rostralmiddlefrontal_1", "L_rostralmiddlefrontal_1", "rostralmiddlefrontal", "frontal"),
            (8, 219, "R", "caudalmiddlefrontal_1", "R_caudalmiddlefrontal_1", "caudalmiddlefrontal", "frontal"),
            (9, 219, "L", "lateraloccipital_1", "L_lateraloccipital_1", "lateraloccipital", "occipital"),
        ],
        dtype=[
            ("node_index", int),
            ("scale", int),
            ("hemi", "U1"),
            ("atlas_label", "U32"),
            ("region_label", "U40"),
            ("base_region", "U32"),
            ("anatomical_group", "U16"),
        ],
    )
    import pandas as pd

    frame = pd.DataFrame(labels)
    adjacency = np.eye(len(frame))
    targets = build_targets(
        adjacency,
        frame,
        target_config={},
        anatomy_config={
            "association_keywords": ["frontal", "parietal", "precuneus", "cingulate", "insula"],
            "occipital_keywords": ["cuneus", "lateraloccipital", "lingual", "pericalcarine", "occipital"],
        },
        atlas_dir="data/external/atlas_cache/atl-cammoun2012/fsaverage5",
    )
    metadata = build_target_metadata(frame, targets)

    assert set(np.flatnonzero(targets["landmark_v1"])) == {0, 1, 2}
    assert np.isclose(np.linalg.norm(targets["landmark_v1"]), 1.0)
    assert metadata.loc[metadata["target_name"].eq("landmark_v1"), "n_support_nodes"].item() == 3
    assert metadata["target_family"].value_counts()["single_cortical_parcel"] == len(frame)


def test_top_k_excluding_nodes_reranks_over_remaining_nodes() -> None:
    scores = np.array([10.0, 8.0, 9.0, 6.0, 7.0])
    selected = top_k_excluding_nodes(scores, k=2, excluded=np.array([0, 2]))
    assert np.array_equal(selected, np.array([1, 4]))
