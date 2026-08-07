#!/usr/bin/env python
"""Manim scene for the reachable-state cloud built from real control results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Axes,
    BLUE,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    LEFT,
    Line,
    ORANGE,
    Polygon,
    RIGHT,
    Scene,
    Text,
    UP,
    VGroup,
    WHITE,
    Dot,
    Arrow,
    Transform,
    LaggedStart,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "figures" / "animation_assets"
DEFAULT_STEM = "reachable_state_cloud_scale-114_k-5_static_sub-014"
COLORS = {
    "degree_strength": ORANGE,
    "cycle_participation": BLUE,
}
TARGET_LABELS = {
    "association_module": "Association module",
    "principal_gradient": "Principal gradient",
    "landmark_lpfc": "lPFC landmark",
}


def load_assets(stem: str = DEFAULT_STEM) -> tuple[dict, np.lib.npyio.NpzFile]:
    with open(ASSET_DIR / f"{stem}.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    arrays = np.load(ASSET_DIR / f"{stem}.npz")
    return metadata, arrays


class ReachableStateCloudScene(Scene):
    def construct(self) -> None:
        metadata, arrays = load_assets()

        boundaries = {
            method: np.asarray(arrays[f"{method}_boundary"], dtype=float)
            for method in ("degree_strength", "cycle_participation")
        }
        boundaries_norm = {
            method: np.asarray(arrays[f"{method}_boundary_normalized"], dtype=float)
            for method in ("degree_strength", "cycle_participation")
        }
        clouds = {
            method: np.asarray(arrays[f"{method}_cloud"], dtype=float)
            for method in ("degree_strength", "cycle_participation")
        }
        clouds_norm = {
            method: np.asarray(arrays[f"{method}_cloud_normalized"], dtype=float)
            for method in ("degree_strength", "cycle_participation")
        }

        all_points = np.vstack(
            [
                boundaries["degree_strength"],
                boundaries["cycle_participation"],
                boundaries_norm["degree_strength"],
                boundaries_norm["cycle_participation"],
                clouds["degree_strength"],
                clouds["cycle_participation"],
                clouds_norm["degree_strength"],
                clouds_norm["cycle_participation"],
            ]
        )
        max_abs = float(np.max(np.abs(all_points))) * 1.18
        axes = Axes(
            x_range=[-max_abs, max_abs, max_abs / 4.0],
            y_range=[-max_abs, max_abs, max_abs / 4.0],
            x_length=8.0,
            y_length=8.0,
            axis_config={"include_ticks": False, "include_numbers": False, "color": "#bbbbbb"},
            tips=False,
        )
        axes_labels = VGroup(
            Text("Projected dimension 1", font_size=24).next_to(axes, DOWN, buff=0.35),
            Text("Projected dimension 2", font_size=24).rotate(np.pi / 2).next_to(axes, LEFT, buff=0.35),
        )

        title = Text(
            f"Reachable-state geometry | scale {metadata['scale']}, k={metadata['k']}",
            font_size=30,
        ).to_edge(UP)
        subtitle = Text(
            f"Representative subject: {metadata['subject_id']}",
            font_size=22,
        ).next_to(title, DOWN, buff=0.15)

        self.play(FadeIn(title), FadeIn(subtitle), FadeIn(axes), FadeIn(axes_labels))

        degree_group = self.make_cloud_group(
            axes=axes,
            boundary=boundaries["degree_strength"],
            cloud=clouds["degree_strength"][::4],
            color=COLORS["degree_strength"],
        )
        degree_metrics = self.metric_block(
            "Degree-based set",
            metadata["methods"]["degree_strength"],
            color=COLORS["degree_strength"],
        ).to_corner(UP + RIGHT).shift(0.35 * DOWN + 0.2 * LEFT)

        self.play(
            LaggedStart(*[FadeIn(mob, scale=0.9) for mob in degree_group], lag_ratio=0.01),
            FadeIn(degree_metrics),
        )
        self.wait(0.6)

        cycle_group = self.make_cloud_group(
            axes=axes,
            boundary=boundaries["cycle_participation"],
            cloud=clouds["cycle_participation"][::4],
            color=COLORS["cycle_participation"],
        )
        cycle_metrics = self.metric_block(
            "Topology-based set",
            metadata["methods"]["cycle_participation"],
            color=COLORS["cycle_participation"],
        ).to_corner(UP + RIGHT).shift(0.35 * DOWN + 0.2 * LEFT)

        self.play(Transform(degree_group, cycle_group), Transform(degree_metrics, cycle_metrics))
        self.wait(0.6)

        note_raw = Text(
            "Raw projection can overemphasize easy directions",
            font_size=22,
        ).to_edge(DOWN).shift(0.3 * UP)
        self.play(FadeIn(note_raw))
        self.wait(0.3)

        degree_norm = self.make_cloud_group(
            axes=axes,
            boundary=boundaries_norm["degree_strength"],
            cloud=clouds_norm["degree_strength"][::4],
            color=COLORS["degree_strength"],
        )
        cycle_norm = self.make_cloud_group(
            axes=axes,
            boundary=boundaries_norm["cycle_participation"],
            cloud=clouds_norm["cycle_participation"][::4],
            color=COLORS["cycle_participation"],
        )
        note_norm = Text(
            "Area-matched view highlights anisotropy and spread",
            font_size=22,
        ).to_edge(DOWN).shift(0.3 * UP)
        self.play(
            Transform(degree_group, degree_norm),
            Transform(cycle_group, cycle_norm),
            Transform(note_raw, note_norm),
        )
        self.wait(0.6)

        degree_overlay = self.make_cloud_group(
            axes=axes,
            boundary=boundaries_norm["degree_strength"],
            cloud=clouds_norm["degree_strength"][::4],
            color=COLORS["degree_strength"],
            fill_opacity=0.12,
            dot_opacity=0.12,
        )
        cycle_overlay = self.make_cloud_group(
            axes=axes,
            boundary=boundaries_norm["cycle_participation"],
            cloud=clouds_norm["cycle_participation"][::4],
            color=COLORS["cycle_participation"],
            fill_opacity=0.12,
            dot_opacity=0.12,
        )
        legend = self.make_legend().to_corner(DOWN + RIGHT).shift(0.15 * LEFT + 0.15 * UP)
        self.play(
            Transform(degree_group, degree_overlay),
            FadeIn(cycle_overlay),
            FadeIn(legend),
            FadeOut(degree_metrics),
            FadeOut(note_raw),
        )
        self.wait(0.5)

        target_arrows = VGroup()
        target_labels = VGroup()
        for target_name in ("association_module", "principal_gradient", "landmark_lpfc"):
            coords = np.asarray(metadata["targets"][target_name]["projection"], dtype=float)
            norm = max(float(np.linalg.norm(coords)), 1.0e-12)
            display = 0.9 * coords / norm * max_abs * 0.75
            arrow = Arrow(
                axes.c2p(0.0, 0.0),
                axes.c2p(display[0], display[1]),
                buff=0.0,
                stroke_width=4,
                color=WHITE,
                max_stroke_width_to_length_ratio=16,
                max_tip_length_to_length_ratio=0.18,
            )
            label = Text(TARGET_LABELS[target_name], font_size=22).next_to(arrow.get_end(), RIGHT if display[0] >= 0 else LEFT, buff=0.12)
            target_arrows.add(arrow)
            target_labels.add(label)

        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in target_arrows], lag_ratio=0.2))
        self.play(FadeIn(target_labels))
        self.wait(1.0)

    def make_cloud_group(
        self,
        axes: Axes,
        boundary: np.ndarray,
        cloud: np.ndarray,
        color,
        fill_opacity: float = 0.2,
        dot_opacity: float = 0.25,
    ) -> VGroup:
        polygon = Polygon(
            *[axes.c2p(point[0], point[1]) for point in boundary],
            stroke_color=color,
            stroke_width=4,
            fill_color=color,
            fill_opacity=fill_opacity,
        )
        dots = VGroup(
            *[
                Dot(axes.c2p(point[0], point[1]), radius=0.015, color=color, fill_opacity=dot_opacity, stroke_opacity=0.0)
                for point in cloud
            ]
        )
        crosshair = VGroup(
            Line(axes.c2p(-10, 0), axes.c2p(10, 0), stroke_color="#666666", stroke_opacity=0.25, stroke_width=1),
            Line(axes.c2p(0, -10), axes.c2p(0, 10), stroke_color="#666666", stroke_opacity=0.25, stroke_width=1),
        )
        return VGroup(crosshair, polygon, dots)

    def metric_block(self, title: str, metrics: dict, color) -> VGroup:
        lines = [
            Text(title, font_size=26, color=color),
            Text(f"log10 energy: {metrics['log10_average_energy']:.3f}", font_size=22),
            Text(f"effective rank: {metrics['effective_rank']:.2f}", font_size=22),
            Text(f"participation ratio: {metrics['participation_ratio']:.2f}", font_size=22),
            Text(f"log10 cond #: {metrics['log10_condition_number']:.2f}", font_size=22),
        ]
        return VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

    def make_legend(self) -> VGroup:
        degree = VGroup(Dot(color=COLORS["degree_strength"], radius=0.08), Text("Degree-based", font_size=22)).arrange(RIGHT, buff=0.12)
        cycle = VGroup(Dot(color=COLORS["cycle_participation"], radius=0.08), Text("Topology-based", font_size=22)).arrange(RIGHT, buff=0.12)
        return VGroup(degree, cycle).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
