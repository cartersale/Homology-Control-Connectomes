*Planning note that motivated the representative-stability check. Schemes 2 and 3 below
are implemented in [`scripts/representative_stability/`](scripts/representative_stability/);
the results are in that directory's `README.md` and `REPORT.md`. Kept here as the record
of what the test was designed to answer and why it was scoped the way it was.*

---

The representative-stability test, concretely. The question is: your cycle-participation scores come from Ripser's cocycle representatives, which are one arbitrary choice among many valid ones for the same $H_1$ feature. Does your headline result — cycle-informed sets have higher effective rank — survive if you compute participation from a *different* representative of the same cycles?

**The logic of the test.** A persistent $H_1$ feature is a homology class; it has a birth, a death, and a persistence, all canonical. What is *not* canonical is which specific cycle (set of edges/vertices) you pick to represent that class. Ripser hands you one via its cohomology reduction. Your `CP_i` score sums persistences over features whose representative *support* includes node $i$ — so if a different-but-equally-valid representative of the same feature touches different nodes, your per-node scores change even though the barcode is identical. The test asks whether that changes the ranking enough to move the downstream control-geometry result.

**The clean version — three representative schemes, same features.**

1. **Ripser cocycles (your current scores).** Baseline.
2. **Length-minimized / geodesic representatives.** For each $H_1$ feature, replace Ripser's representative with a shorter cycle in the same class. For each feature, take the birth simplex and find the shortest cycle through it in the graph at the death threshold. This gives geodesic-ish representatives that are localized where Ripser's are not.
3. **A different library's representatives.** `ripser.py` / `gudhi` / `dionysus` don't all return the same cocycles. Recomputing the same filtration in a second package and re-deriving `CP_i` is the least-work way to get an *independent* representative choice you didn't hand-craft — which is exactly the "implementation dependence" your limitation names.

For each scheme, recompute `CP_i`, re-rank, re-select the top-$k$ cycle-participation driver sets, recompute effective rank / participation ratio, and ask: **does the cycle-minus-degree effective-rank advantage survive?**

**What to actually measure — three levels, escalating in what they'd cost you if they fail.**

- *Score level:* Spearman correlation between `CP_i` under scheme 1 vs 2 vs 3, per subject per scale. If these are ~0.9+, representatives barely matter and you're done — report the correlation and move on. If they're ~0.5, the scores are genuinely unstable and you need the next two levels.
- *Selection level:* overlap of the top-$k$ driver sets across schemes (Jaccard, or just "how many of the top-15 are shared"). This is what actually feeds the control analysis — scores can wobble while the *top* set stays stable, which would be a reassuring result.
- *Result level:* the one that matters. Recompute the cycle-vs-degree effective-rank difference (your Fig. 4 headline) under each scheme. If the advantage is still positive and similar in magnitude across all three, your result is representative-robust and you say so in one sentence. If it collapses under scheme 2 or 3, you've found that the headline is an artifact of Ripser's particular representatives — which you'd need to know before the *journal* submission, not just ComPer.

**Scope it to fit the time.** Don't run all 70 subjects × 3 scales × 3 schemes. Do **one scale (114) × all 70 subjects × schemes 1 and 3** first — that's the cheapest independent-representative comparison (two library calls, no custom geometry) and it answers the core question. Only build scheme 2 (geodesic) if 1-vs-3 already shows instability and you need to understand it. One scale, one extra library, the three-level readout — that's a few hours, and it's the version that both pre-empts the ComPer question and strengthens the paper.