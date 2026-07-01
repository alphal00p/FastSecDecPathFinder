# FSD Integration Report

## Run

- result file: `/Users/vjhirsch/Documents/Work/pygloop/FSD_v2/examples/outputs/example_hard_from_polynomial_all_raw_results.json`
- run card: `/Users/vjhirsch/Documents/Work/pygloop/FSD_v2/examples/runs/four_loop_hard_all_sectors_fsd_qmc.toml`
- integral: `uf`
- prefactor convention: `sector`
- sampling mode: `qmc`
- QMC backend: `pysecdec-default`
- QMC support grouping: `boundary`
- QMC optimized evaluator groups: `3728`
- sectors sampled: `3728` / `3728`
- total raw samples: `258,198,244,544`
- elapsed integration wall time: `7777.277 s`
- average eval time: `0.1279 us/sample/worker`
- optimized-QMC fallback sectors: `37, 104, 153, 921, 1590, 1725`
- refined sectors for per-sector `1e-3` target: `138, 163, 189, 656, 772, 919, 1461, 1722, 2208`
- refinement raw samples per selected sector: `1,060,603,168`

## Generation

- total recorded generation time: `53.308 s`

| stage | time | detail |
|---|---:|---|
| U/F input load | 540 us | - |
| U/F input polynomial construction | 57.6 ms | - |
| U/F extraction | 182 us | - |
| Symbolica scalar evaluator build | 613 us | - |
| pySecDec sector symmetry summary | 0 us | raw=3728, disabled |
| pySecDec sector decomposition | 6.599 s | geometric_infinity_no_primary |
| FSD SectorDefinition conversion | 3.076 s | 3728 sectors |
| Symbolica sector evaluator build skipped | 0 us | explicit backend uses sector-level evaluators directly |
| Symbolica sector evaluator build | 2.79 us | 3728 sectors |
| Symbolica Taylor evaluator build | 0 us | skipped: sector evaluator backend prepares explicit sector integrand evaluators |
| Symbolica subtraction formula build | 2.65 ms | 7 endpoint projector signature(s), 0 regular Taylor signature(s), endpoint cache hits/generated=4/0 |
| Symbolica explicit sector build | 43.572 s | 3728 single-evaluator sector integrand(s) |

## Aggregate Laurent Sum

The aggregate coefficients below are the total sector sum stored in the result JSON.  In independent-sector QMC mode the reported errors are the quadrature propagation of per-sector errors.

| order | value | MC error | relative error |
|---|---:|---:|---:|
| eps^-8 | 0 | 0 | n/a |
| eps^-7 | 0 | 0 | n/a |
| eps^-6 | 0 | 0 | n/a |
| eps^-5 | 0 | 0 | n/a |
| eps^-4 | 0 | 0 | n/a |
| eps^-3 | 0 | 0 | n/a |
| eps^-2 | -3.60617208198 | 9.72838068827e-07 | 2.698e-07 |
| eps^-1 | -16.6719719507 | 6.83942135807e-06 | 4.102e-07 |
| eps^0 | -149.867746145 | 5.92308205166e-05 | 3.952e-07 |

## Per-Sector Runtime

| statistic | value | sector |
|---|---:|---|
| min | 0.04781 us/sample | PSD1037 (1037) |
| max | 0.2178 us/sample | PSD2779 (2779) |
| average | 0.1283 us/sample | - |
| median | 0.1391 us/sample | - |

## Per-Sector Relative Accuracy

- threshold checked: `1.0e-03`
- sectors above threshold: `0` / `3728`

| statistic | max relative error | sector/order |
|---|---:|---|
| min sector max | 2.313e-06 | PSD1104 (1104), eps^-1 |
| max sector max | 0.0009829 | PSD3359 (3359), eps^-1 |
| average sector max | 1.524e-05 | - |
| median sector max | 9.892e-06 | - |

| worst sector | max relative error | order | samples |
|---|---:|---|---:|
| PSD3359 (3359) | 0.0009829 | eps^-1 | 66,860,128 |
| PSD2026 (2026) | 0.0009597 | eps^0 | 66,860,128 |
| PSD3430 (3430) | 0.0008832 | eps^-1 | 66,860,128 |
| PSD3286 (3286) | 0.0007289 | eps^-1 | 66,860,128 |
| PSD3302 (3302) | 0.0005997 | eps^-1 | 66,860,128 |
| PSD3506 (3506) | 0.0005946 | eps^-1 | 66,860,128 |
| PSD3586 (3586) | 0.0005213 | eps^-1 | 66,860,128 |
| PSD3632 (3632) | 0.0005124 | eps^-1 | 66,860,128 |
| PSD3403 (3403) | 0.0004963 | eps^-1 | 66,860,128 |
| PSD3597 (3597) | 0.0004905 | eps^-1 | 66,860,128 |
| PSD3667 (3667) | 0.0004537 | eps^-1 | 66,860,128 |
| PSD3268 (3268) | 0.0004423 | eps^-1 | 66,860,128 |
| PSD2845 (2845) | 0.0004411 | eps^0 | 66,860,128 |
| PSD3279 (3279) | 0.0004348 | eps^-1 | 66,860,128 |
| PSD3055 (3055) | 0.0004037 | eps^0 | 66,860,128 |
| PSD3612 (3612) | 0.0003826 | eps^-1 | 66,860,128 |
| PSD3581 (3581) | 0.0003605 | eps^-1 | 66,860,128 |
| PSD3590 (3590) | 0.0003427 | eps^-1 | 66,860,128 |
| PSD2579 (2579) | 0.0002972 | eps^0 | 66,860,128 |
| PSD3325 (3325) | 0.0002834 | eps^-1 | 66,860,128 |

## Weight And Precision Diagnostics

- maximum absolute sampled weight: `1188.72` in PSD3322 (3322)
- ordinary: `258,198,244,542` samples (100%)
- stability: `0` samples (0%)
- medium_precision: `0` samples (0%)
- high_precision: `2` samples (7.75e-10%)

