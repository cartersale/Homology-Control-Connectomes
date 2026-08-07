#!/usr/bin/env python
"""Minimal Manim animation for effective rank, participation ratio, and condition number."""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DecimalNumber,
    Dot,
    DOWN,
    Ellipse,
    FadeIn,
    FadeOut,
    GREEN,
    LEFT,
    Line,
    MathTex,
    ORANGE,
    RIGHT,
    Scene,
    UP,
    ValueTracker,
    VGroup,
    WHITE,
    always_redraw,
    config,
)


config.background_color = "#111318"


class GeometricMeasuresScene(Scene):
    def construct(self) -> None:
        lam_1 = ValueTracker(2.4)
        lam_2 = ValueTracker(2.4)

        axes = VGroup(
            Line(3.1 * LEFT, 3.1 * RIGHT, color="#8b93a7", stroke_width=2),
            Line(2.35 * DOWN, 2.35 * UP, color="#8b93a7", stroke_width=2),
        ).shift(1.2 * LEFT)
        labels = VGroup(
            MathTex(r"x_1", color="#c9ceda", font_size=34).next_to(axes[0], RIGHT, buff=0.12),
            MathTex(r"x_2", color="#c9ceda", font_size=34).next_to(axes[1], UP, buff=0.12),
        )

        ellipse = always_redraw(
            lambda: self.ellipse_from_lambdas(
                lam_1.get_value(),
                lam_2.get_value(),
                center=axes.get_center(),
            )
        )
        principal_axes = always_redraw(
            lambda: self.principal_axes(
                lam_1.get_value(),
                lam_2.get_value(),
                center=axes.get_center(),
            )
        )
        spectrum = self.spectrum_bars(lam_1, lam_2).to_edge(RIGHT).shift(1.5 * DOWN)
        equations = self.equation_block(lam_1, lam_2).to_edge(RIGHT).shift(0.75 * UP)

        self.play(FadeIn(axes), FadeIn(labels))
        self.play(FadeIn(ellipse), FadeIn(principal_axes), FadeIn(spectrum), FadeIn(equations))
        self.wait(0.8)

        for high, low in ((4.0, 1.6), (4.8, 0.85), (5.4, 0.22), (2.8, 2.8)):
            self.play(
                lam_1.animate.set_value(high),
                lam_2.animate.set_value(low),
                run_time=2.0,
            )
            self.wait(0.45)

        self.play(FadeOut(ellipse), FadeOut(principal_axes), FadeOut(spectrum), FadeOut(equations), FadeOut(axes), FadeOut(labels))

    @staticmethod
    def ellipse_from_lambdas(lam_1: float, lam_2: float, center: np.ndarray) -> Ellipse:
        width = 1.76 * np.sqrt(lam_1)
        height = 1.76 * np.sqrt(lam_2)
        return Ellipse(width=width, height=height, color=BLUE, stroke_width=5).move_to(center)

    @staticmethod
    def principal_axes(lam_1: float, lam_2: float, center: np.ndarray) -> VGroup:
        major = 0.88 * np.sqrt(lam_1)
        minor = 0.88 * np.sqrt(lam_2)
        return VGroup(
            Line(center + major * LEFT, center + major * RIGHT, color=ORANGE, stroke_width=5),
            Line(center + minor * DOWN, center + minor * UP, color=GREEN, stroke_width=5),
        )

    @staticmethod
    def spectrum_bars(lam_1: ValueTracker, lam_2: ValueTracker) -> VGroup:
        max_lam = 5.6
        rows = VGroup()
        for tracker, label, color in ((lam_1, r"\lambda_1", ORANGE), (lam_2, r"\lambda_2", GREEN)):
            anchor = Dot(color=color).set_opacity(0.0)
            tex = MathTex(label, color=WHITE, font_size=30).next_to(anchor, LEFT, buff=0.22)
            bar = always_redraw(
                lambda a=anchor, t=tracker, c=color: Line(
                    a.get_center(),
                    a.get_center() + (2.0 * t.get_value() / max_lam) * RIGHT,
                    color=c,
                    stroke_width=12,
                )
            )
            rows.add(VGroup(tex, anchor, bar))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.36)
        return rows

    @staticmethod
    def equation_block(lam_1: ValueTracker, lam_2: ValueTracker) -> VGroup:
        def metric_values() -> tuple[float, float, float]:
            values = np.array([lam_1.get_value(), lam_2.get_value()], dtype=float)
            probabilities = values / values.sum()
            entropy = -float(np.sum(probabilities * np.log(probabilities)))
            effective_rank = float(np.exp(entropy))
            participation_ratio = values.sum() ** 2 / float(np.sum(values**2))
            condition = float(values.max() / values.min())
            return effective_rank, participation_ratio, condition

        effective_rank, participation_ratio, condition = metric_values()
        value_mobs = (
            DecimalNumber(effective_rank, num_decimal_places=2, font_size=34),
            DecimalNumber(participation_ratio, num_decimal_places=2, font_size=34),
            DecimalNumber(condition, num_decimal_places=2, font_size=34),
        )

        for index, number in enumerate(value_mobs):
            number.add_updater(lambda mob, i=index: mob.set_value(metric_values()[i]))

        rows = VGroup(
            VGroup(MathTex(r"r_{\mathrm{eff}}=e^{H(p)}", font_size=34), value_mobs[0]),
            VGroup(MathTex(r"\mathrm{PR}={(\sum_i\lambda_i)^2\over\sum_i\lambda_i^2}", font_size=34), value_mobs[1]),
            VGroup(MathTex(r"\kappa={\lambda_{\max}\over\lambda_{\min}}", font_size=34), value_mobs[2]),
        )
        for row in rows:
            row.arrange(RIGHT, buff=0.22)
            row.set_color(WHITE)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        return rows

    @staticmethod
    def metric_values_from_lambdas(lam_1: float, lam_2: float) -> tuple[float, float, float]:
        values = np.array([lam_1, lam_2], dtype=float)
        probabilities = values / values.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        effective_rank = float(np.exp(entropy))
        participation_ratio = values.sum() ** 2 / float(np.sum(values**2))
        condition = float(values.max() / values.min())
        return effective_rank, participation_ratio, condition
