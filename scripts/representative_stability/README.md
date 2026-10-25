# Representative-stability test for cycle participation

Standalone check, kept out of the main `scripts/` pipeline: does the cycle-informed
effective-rank advantage survive if `CP_i` is computed from a **different
representative of the same H1 features**?

A persistent H1 feature is a homology class. Its birth, death, and persistence are
canonical; the cycle chosen to represent it is not. The main pipeline
(`hoc.topology.cycle_participation`) scores nodes from the support of Ripser's
cocycle representatives, so a different-but-equally-valid representative can shift
every per-node score while leaving the barcode identical.

This directory implements two alternative representative schemes — length-minimized
(geodesic) representatives and an independent library's representatives (GUDHI) —
and runs the three-level readout against the Ripser baseline.

## Files

| File | Purpose |
| --- | --- |
| `representative_schemes.py` | Representative construction: `ripser` (baseline), `geodesic` (length-minimized), `gudhi` (second library's own pairing simplices), `gudhi_geodesic` (length-minimized cycle on GUDHI's birth edge). Also `compare_libraries`, which checks Ripser and GUDHI against each other. |
| `run_representative_stability.py` | Runs one scale × all subjects × both schemes and writes the three-level readout. |
| `analyze_selection_anatomy.py` | Follow-up: where the two schemes' driver sets land anatomically and spatially. Reads the stability-run outputs, recomputes no topology. |
| `test_representative_schemes.py` | Unit tests for the representative construction. |

## Method — scheme 2 (geodesic)

For each finite H1 feature:

1. Identify the **birth edge** `(u, v)` as the cocycle-support edge whose filtration
   value equals the feature's birth. Ripser computes in single precision, so this
   match is made with a float32-scale tolerance, not float64 epsilon.
2. Delete `(u, v)` and take the shortest `u → v` path in the thresholded graph.
   Path plus birth edge is the length-minimized representative cycle.
3. Score `CP_i` exactly as the main pipeline does — sum persistence over the nodes
   in the representative's support, then min-max normalize within subject.

Two knobs:

- `--weight-mode {filtration,hops}` — length as summed edge distance (geodesic,
  default) or edge count (combinatorially shortest).
- `--threshold-mode {birth,below_death,death}` — which filtration stage the search
  runs in. **Default is `birth`, not the graph at the death threshold.** Vietoris-Rips
  is a *flag* complex: a cycle is filled as soon as its own diameter is reached, so
  searching at the death threshold routinely
  returns a shortest cycle that already bounds — homologous to zero, and not a
  representative of the class at all. On scale-114 subject 0, searching at `death`
  makes 110 of 115 representatives bounding triangles; `below_death` still makes 64
  of 115 bounding. At the birth threshold the failure is impossible: the class is
  created by inserting `(u, v)`, so every cycle through `(u, v)` there is nonzero in
  `H_1`. The diagnostics table reports `n_bounding_representatives` so this is
  auditable, and `below_death` remains available for comparison.

Cycles whose birth edge cannot be located, or whose endpoints are disconnected in
the thresholded graph, fall back to Ripser's support and are counted separately. On
this dataset neither fallback ever fires.

## Method — scheme 3 (GUDHI)

The same Rips filtration is rebuilt in GUDHI (`RipsComplex` → 2-skeleton →
`compute_persistence`), which reduces in **homology** rather than Ripser's
cohomology. `flag_persistence_generators()` returns, per H1 class, the edge that
created it and the edge whose triangle destroyed it.

- `gudhi` scores the vertices of those two pairing simplices — the support GUDHI
  itself hands back, with no post-processing.
- `gudhi_geodesic` instead builds the scheme-2 cycle on GUDHI's birth edge, which
  separates disagreement about the *birth simplex* from disagreement about the
  representative constructed on top of it.

`compare_libraries()` matches features across libraries by their (birth, death)
coordinates and reports barcode agreement plus birth-edge agreement. GUDHI is
imported lazily, so the Ripser-side schemes still run without it (`pip install
gudhi`).

## Running

```bash
# default: scale 114, all 70 subjects, ripser + geodesic + gudhi (~30 s)
python scripts/representative_stability/run_representative_stability.py

# add the fourth scheme that isolates the birth-edge choice
python scripts/representative_stability/run_representative_stability.py \
    --schemes ripser geodesic gudhi gudhi_geodesic

# comparison run at the death-side threshold, into its own results subdirectory
python scripts/representative_stability/run_representative_stability.py \
    --threshold-mode below_death --output-subdir representative_stability_below_death

# tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest scripts/representative_stability/
```

Outputs go to `results/<output-subdir>/`, nothing under `results/primary/` is
touched. Per-subject tables (`node_scores_by_scheme`, `score_correlations`,
`selection_overlap`, `geometry_by_scheme`, `representative_diagnostics`) sit
alongside their `*_summary` counterparts; `run_settings_scale-*.csv` records the
configuration the run used.

## Readouts

0. **Library level** (GUDHI only) — barcode and birth-edge agreement between Ripser
   and GUDHI, per subject.
1. **Score level** — Spearman correlation of `CP_i` between every scheme pair, per
   subject.
2. **Selection level** — Jaccard and shared-node count of the top-k driver sets
   (`k ∈ {1, 2, 3, 5, 15}`), scheme vs scheme and each scheme vs degree.
3. **Result level** — the cycle-minus-degree difference in effective rank (and
   participation ratio, average energy, condition number, minimum eigenvalue) under
   each scheme, with bootstrap median CIs and Wilcoxon signed-rank tests;
   `scheme_contrast_summary` additionally tests whether that advantage *itself*
   differs between schemes.

## Result on scale 114, n = 70 (default birth-threshold run)

### Scheme 3 — GUDHI

- **The two libraries agree completely on the features and on the birth simplex.**
  Across all 70 subjects: same H1 feature count, barcodes matching to 3e-8 (Ripser's
  float32 rounding, nothing more), and **the same birth edge for 100 % of matched
  features** — 7 866 of 7 866, every subject at 1.000. `gudhi_geodesic` is therefore
  bit-identical to `geodesic` (Spearman 1.000, chamfer 0 mm), which is the cleanest
  possible statement of the result: the two implementations make the *same* choice of
  birth simplex, so the "implementation dependence" the limitation names does not
  live there.
- **What does differ is what each library hands back as a representative.** GUDHI
  returns the birth/death *pairing* simplices (mean support 3.2 nodes); Ripser
  returns a cocycle support (5.9); the geodesic construction returns an actual cycle
  (5.1). Median Spearman between Ripser and GUDHI `CP_i` is **0.54** (CI 0.52–0.57),
  between GUDHI and geodesic **0.68**.
- **Result level: the advantage survives under GUDHI too, and sits between the other
  two schemes.** Median cycle-minus-degree effective rank +0.55 (k = 2), +0.96
  (k = 3), +1.57 (k = 5), +2.62 (k = 15), all p ≤ 1.7e-11. Against Ripser the paired
  contrast is indistinguishable at k = 2
  (p = 0.12) and negative from k = 3 up.
- Anatomically GUDHI's selections sit between the two: lobe-composition cosine 0.86
  vs Ripser and 0.95 vs geodesic; chamfer 17.8 mm vs Ripser and 12.5 mm vs geodesic
  (random-set null 27.0 mm).

### Scheme 2 — geodesic

- **Score level: unstable.** Median Spearman(Ripser, geodesic) = 0.40
  (95% CI 0.36–0.44, range 0.11–0.65). Well below the ~0.9 that would have ended the
  test at level 1, so levels 2 and 3 carry the answer.
- **Selection level: largely disjoint.** At k = 5, the two schemes share on average
  0.86 of 5 nodes (mean Jaccard 0.10); at k = 15, 4.8 of 15 (0.20). The geodesic
  scheme also lands closer to degree than Ripser's does (mean Jaccard vs degree 0.46
  vs 0.09 at k = 5) — length-minimized representatives concentrate on well-connected
  nodes.
- **Result level: the advantage survives, but shrinks.** Median cycle-minus-degree
  effective rank stays positive and significant under the geodesic scheme at every
  k ≥ 2 (k = 2: +0.02, p = 2e-3; k = 3: +0.16, p = 8e-6; k = 5: +0.33, p = 2e-8;
  k = 15: +0.53, p = 3e-8), against +0.44, +1.25, +2.46, +5.75 under Ripser. The
  paired contrast is significantly negative for k ≥ 3, i.e. the shrinkage is real
  and not noise. At k = 1 the Ripser advantage is *negative* (−0.11) while the
  geodesic one is ~0.
- The `below_death` comparison run reproduces the same qualitative pattern
  (k = 5: +0.76; k = 15: +1.23), so the conclusion is not an artifact of the
  threshold choice.

## Where the selections land (`analyze_selection_anatomy.py`)

Disjoint node identities need not mean disjoint anatomy: two schemes can pick
different subparcels of the same gyrus and share zero indices. This script tests
that at four granularities — node identity, base region, lobe/hemisphere
composition (enrichment against the parcel-count null, chi-square between arms),
and physical proximity via chamfer distance between driver-set centroids on the
fsaverage5 pial surface, referenced to a random-set null.

```bash
python scripts/representative_stability/analyze_selection_anatomy.py --k-values 5 15
```

Outputs land in `results/<results-subdir>/anatomy/`. Findings at scale 114, n = 70,
k = 15:

- **Same broad territory, different emphasis.** Both schemes are frontal-and-
  cingulate dominated, but the geodesic scheme is far more so (cingulate enrichment
  2.9 vs 0.8; frontal 1.7 vs 1.2) while Ripser's spreads into parietal (1.27 vs
  0.63), occipital (1.02 vs 0.31), and temporal cortex (0.43 vs 0.16). The lobe
  compositions differ significantly (χ² = 242, p = 2e-49, Cramér's V = 0.34), and
  Ripser's shows a right-hemisphere bias the geodesic scheme does not (p = 4e-5).
- **The geodesic scheme is close to degree, Ripser is not.** Median per-subject lobe
  cosine 0.96 (geodesic vs degree) against 0.78 (Ripser vs geodesic) and 0.69
  (Ripser vs degree); median shared nodes 0.67 vs 0.33 and 0.27 of 15. Its lobe
  enrichment profile tracks degree's almost rank for rank.
- **Spatially closer than chance, but not co-located.** Median chamfer distance
  between the Ripser and geodesic sets is 20.5 mm against a 27.0 mm random-set null
  (p = 3e-11) — the two schemes agree on the general neighbourhood far better than
  chance, and much better than node-identity overlap alone suggests. Geodesic vs
  degree is 9.8 mm, roughly the width of one parcel.
- **Whole-score view agrees.** Cohort-mean `CP_i` per lobe under the geodesic scheme
  concentrates in cingulate/insula/frontal and drops sharply in temporal and
  occipital; Ripser's is much flatter across lobes (0.30–0.47 everywhere).

*Caveat on grouping.* `hoc.atlas.classify_region_group` matches `cuneus` before
`precuneus`, so all four precuneus parcels are counted as **occipital** rather than
parietal. They carry 53 of Ripser's 160 occipital selections at k = 15. Correcting
that would move mass from occipital to parietal in both arms without changing the
direction of any contrast above, but the same mislabel affects the main pipeline's
anatomical tables.

**Reading (anatomy).** The two schemes are not picking unrelated brain regions —
they overlap in frontal/cingulate cortex and sit closer together than chance in
space. What length minimization does is pull the score toward *hub-like,* midline,
well-connected territory, i.e. toward degree, and away from the parietal, occipital,
and temporal nodes that Ripser's cocycle supports single out. That is a coherent
mechanism for the shrunken effect size: the more a cycle representative concentrates
on high-degree nodes, the less distinct its driver set is from the degree baseline.

**Reading.** The headline direction is representative-robust — cycle-informed driver
sets beat degree-matched ones on effective rank under *all three* representative
schemes, for every k ≥ 2 — but the *magnitude* is representative-dependent, losing
roughly a factor of 2 under GUDHI's pairing simplices and 4–10 under length-minimized
representatives. The honest one-sentence claim is about sign and significance, not
effect size.

Two library-level statements are now available and worth separating in the
manuscript. Ripser and GUDHI agree on the barcode *and* on the birth simplex of every
feature, so the result does not depend on which package computes the persistence.
What it does depend on is the convention for turning a feature into a node set —
cocycle support, pairing simplices, or an explicit shortest cycle — and that
convention is a modelling choice the paper makes, not an implementation detail it
inherits.
