# Rescued supplement-table builders (frozen, do not edit)

These scripts built every `supp_tables/*.tex` file of the NETN resubmission on
2026-10-19. They were written in a temporary session folder and are copied here
unchanged so that the tables in the revision draft can be traced to code.

- They use absolute paths to the author's machine and to the temporary folder.
  They are kept as a record, not as runnable code.
- `tables/index.txt` and `work/log.txt` are the original run logs; the
  `*_notes.txt` files are the builders' own notes. The combined `.tex` outputs
  are kept in the local baseline archive (see `baseline/README.md`).
- The step that split the combined outputs into individual `S*_*.tex` files was
  never scripted, and `S_canonical.tex` was written by hand.

Runnable ports live in `scripts/paper/` (issue #3).

| Builder | Tables |
| --- | --- |
| `work/s4.py` | S4_1–S4_5 |
| `work/s5.py` | S5_1–S5_11 |
| `work/s6.py` | S6_1–S6_6 |
| `scripts/make_s7.py` (+ `scripts/ls_q1.py`) | S7_1–S7_11 |
| `make_s8.py` | S8_1–S8_10 |
| `work/s9.py` | S9_1–S9_8 |
| `scripts/make_s10_s11.py` | S10_*, S11_* |
| `a1.py`, `a2.py` | inline numbers in main.tex L135 (no table) |
