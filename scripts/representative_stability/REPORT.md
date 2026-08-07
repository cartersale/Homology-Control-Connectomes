# Representative stability of cycle participation

**Question.** Cycle-participation scores `CP_i` are computed from Ripser's H1 cocycle
representatives, which are one arbitrary choice among many valid representatives of
the same homology class. Does the headline result — cycle-informed driver sets have
higher effective rank than degree-matched ones — survive if `CP_i` is derived from a
different representative of the *same* features?

**Answer.** Yes in sign and significance, no in magnitude. The advantage is positive
and significant for every k ≥ 2 under all three representative schemes, but its size
varies by a factor of 2–10 across them. Separately, Ripser and GUDHI are shown to
agree on the barcode *and* on the birth simplex of every feature, so the variation is
attributable to the convention for turning a feature into a node set, not to the
persistence software.

Scale 114, all 70 subjects, primary control parameters (`c = 0.1`, `ε = 1e-5`).
Code and outputs: `scripts/representative_stability/`, `results/representative_stability/`.

---

## Methods

All three schemes score nodes identically once a representative is chosen: `CP_i` is
the sum of persistence over features whose representative support contains node *i*,
min–max normalized within subject, exactly as in `hoc.topology.cycle_participation`.
Only the representative differs.

**Scheme 1 — Ripser cocycles (baseline).** The pipeline's current definition: support
of the H1 cocycles returned by `ripser.py` under its cohomology reduction.

**Scheme 2 — length-minimized (geodesic).** For each finite H1 feature, the birth edge
`(u, v)` is identified as the cocycle-support edge whose filtration value equals the
feature's birth; the edge is deleted and the shortest `u → v` path is taken in the
thresholded graph, so path plus birth edge is a length-minimized cycle for that class.
Two implementation points matter. (i) Ripser reduces in single precision, so birth
values must be matched with a float32-scale tolerance; with float64 tolerance ~96 % of
birth edges fail to match. (ii) The search runs at the **birth** threshold, not the
death threshold suggested by the original plan. Vietoris–Rips is a flag complex, so a
cycle is filled once its own diameter is reached: at the death threshold the shortest
cycle is usually a filled triangle, homologous to zero and not a representative at all
(110 of 115 features on one subject; 64 of 115 just below death). At the birth
threshold this is impossible, because the class is created by inserting `(u, v)`.
A `below_death` comparison run is reported alongside and reproduces the conclusions.

**Scheme 3 — GUDHI (independent library).** The same filtration is rebuilt in GUDHI,
which reduces in homology rather than cohomology. `flag_persistence_generators()`
returns the birth edge and death edge pairing each H1 class; the union of their
vertices is the representative support, with no post-processing. A fourth diagnostic
arm, `gudhi_geodesic`, applies the scheme-2 construction to GUDHI's birth edge, which
separates disagreement about the *birth simplex* from disagreement about the
representative built on it.

**Readouts.** (0) Library agreement: features matched across libraries by their
canonical (birth, death) coordinates; barcode offset and birth-edge identity.
(1) Score level: per-subject Spearman correlation of `CP_i` between schemes.
(2) Selection level: overlap of top-k driver sets (k ∈ {1, 2, 3, 5, 15}).
(3) Result level: cycle-minus-degree difference in Gramian effective rank per subject,
summarized by median, bootstrap 95 % CI (2 000 resamples), and two-sided Wilcoxon
signed-rank test; a paired contrast tests whether that advantage itself differs
between schemes. An anatomical follow-up compares lobe composition, region-level
agreement, and centroid chamfer distance on the fsaverage5 surface against a
random-set null.

---

## Results

### 0. The two libraries agree completely

| quantity | value |
| --- | --- |
| H1 features, Ripser vs GUDHI | 112.4 per subject, identical counts |
| maximum barcode offset | 3.0e-8 (Ripser's float32 rounding) |
| features with the same birth edge | **7 866 / 7 866 (100 %, every subject at 1.000)** |

Consequently `gudhi_geodesic` is identical to `geodesic` (Spearman 1.000, chamfer
0 mm). The birth simplex is not an implementation-dependent choice.

### 1. Scores diverge substantially

Median per-subject Spearman correlation of `CP_i` (95 % CI):

| pair | median ρ | 95 % CI | range |
| --- | --- | --- | --- |
| Ripser vs GUDHI | 0.54 | 0.52 – 0.57 | 0.30 – 0.76 |
| Ripser vs geodesic | 0.40 | 0.36 – 0.44 | 0.11 – 0.65 |
| GUDHI vs geodesic | 0.68 | 0.65 – 0.70 | 0.53 – 0.81 |

All are far below the ~0.9 that would have made the question moot, so the selection
and result levels carry the answer. The schemes differ in what they return: mean
support size 5.9 nodes (Ripser cocycle), 3.2 (GUDHI pairing simplices), 5.1 (explicit
shortest cycle).

### 2. Driver sets overlap weakly, and the alternatives drift toward degree

Mean Jaccard of top-k sets:

| pair | k = 5 | k = 15 |
| --- | --- | --- |
| Ripser vs geodesic | 0.10 | 0.20 |
| Ripser vs GUDHI | 0.19 | 0.27 |
| geodesic vs degree | 0.46 | 0.50 |
| GUDHI vs degree | 0.21 | 0.30 |
| Ripser vs degree | 0.09 | 0.18 |

### 3. The effective-rank advantage survives under all three schemes

Median cycle-minus-degree effective rank across 70 subjects (Wilcoxon *p*):

| k | Ripser | GUDHI | geodesic |
| --- | --- | --- | --- |
| 1 | −0.11 (3e-3) | +0.13 (3e-4) | 0.00 (4e-4) |
| 2 | +0.44 (5e-11) | +0.55 (2e-11) | +0.02 (2e-3) |
| 3 | +1.25 (5e-13) | +0.96 (5e-13) | +0.16 (8e-6) |
| 5 | +2.46 (4e-13) | +1.57 (4e-13) | +0.33 (2e-8) |
| 15 | +5.74 (4e-13) | +2.62 (4e-13) | +0.53 (3e-8) |

Participation ratio behaves the same way (k = 15: +6.28, +2.01, +0.29). The paired
scheme contrast is not significant for GUDHI vs Ripser at k = 2 (p = 0.12) and turns
negative from k = 3 up; for geodesic vs Ripser it is negative from k = 2 up. At k = 1
the schemes disagree in sign — Ripser's single-node advantage is negative — and many
subjects contribute exact zeros (25 of 70 for geodesic, where the cycle and degree
choice coincide), so the k = 1 tests are effectively over a subset of the cohort and
should not be read as a cohort-level effect.

### 4. Where the selections land

The schemes are not selecting unrelated regions. Median chamfer distance between
top-15 driver-set centroids is 20.5 mm (Ripser vs geodesic) and 17.8 mm (Ripser vs
GUDHI) against a 27.0 mm random-set null (p ≤ 3e-11), and lobe-composition cosine is
0.78 and 0.86 respectively. But the emphasis shifts systematically (χ² over lobes,
Ripser vs geodesic: 242, p = 2e-49):

| lobe | Ripser | GUDHI | geodesic | degree |
| --- | --- | --- | --- | --- |
| cingulate | 0.80 | 1.95 | 2.92 | 3.47 |
| frontal | 1.20 | 1.68 | 1.67 | 1.89 |
| parietal | 1.27 | 0.86 | 0.63 | 0.21 |
| occipital | 1.02 | 0.45 | 0.31 | 0.30 |
| temporal | 0.43 | 0.22 | 0.16 | 0.05 |

(Enrichment relative to parcel counts. Note that `hoc.atlas.classify_region_group`
matches `cuneus` before `precuneus`, so the four precuneus parcels are counted as
occipital; correcting this shifts mass from occipital to parietal in every column
without changing the ordering.)

The alternative representatives concentrate on midline, well-connected territory —
i.e. toward degree — while Ripser's cocycle supports retain more parietal, occipital,
and temporal mass. This is a plausible mechanism for the shrinking effect size: the
more a representative concentrates on high-degree nodes, the less its driver set
differs from the degree baseline it is being compared against.

---

## Conclusion

The cycle-versus-degree effective-rank advantage is representative-robust in
direction: positive and significant for every k ≥ 2 under Ripser cocycles, GUDHI
pairing simplices, and length-minimized cycles alike. It is not representative-robust
in magnitude, ranging from +0.53 to +5.74 at k = 15 depending on the scheme. Two
statements should therefore be kept separate in the manuscript:

1. **Not software-dependent.** Ripser and GUDHI produce the same barcode and the same
   birth simplex for every feature in the cohort, so nothing here depends on which
   package computes the persistence.
2. **Convention-dependent in size.** How a feature is converted into a node set —
   cocycle support, pairing simplices, or an explicit shortest cycle — is a modelling
   choice, and it sets the effect size. The defensible claim is about sign and
   significance; any quoted magnitude should be labelled as specific to the
   cocycle-support definition.

Remaining scope: this test covers one parcellation scale (114). Repeating it at 68 and
219 is inexpensive (~30 s per scale) and would establish whether the ordering of
schemes is scale-stable.
