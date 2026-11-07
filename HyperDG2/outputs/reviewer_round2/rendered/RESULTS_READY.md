# Final reviewer-round results

This report is rendered only after the finalized analysis records 384/384 complete runs (six conditions, four datasets, four targets, and four seeds). Source-validation selection excludes the held-out target; target-domain accuracy is read only after selection. Effects are percentage-point treatment-minus-control differences.

The held-out target is the inferential cluster for each dataset. Dataset estimates average four paired run-seed differences within each of four named targets. The suite primary analysis uses the four dataset means as its inference clusters while retaining all 16 target-domain contributions. Non-significance is not interpreted as equivalence.

## Primary suite-level conclusions

- **Full per-domain - Standard ERM:** -1.61 pp (95% CI [-3.17, -0.04]; Holm-adjusted p=0.250). Evidence relative to zero: no adjusted evidence against zero. Practical-effect classification: suite macro equivalence not demonstrated.
- **No-variance per-domain - Shared learnable:** -0.13 pp (95% CI [-0.53, 0.27]; Holm-adjusted p=0.375). Evidence relative to zero: no adjusted evidence against zero. Practical-effect classification: suite macro equivalent within +/-1.0 pp.

## Six-condition source-selected summary

Accuracy is the equal-target macro mean. The two SD columns keep between-target and within-target run-seed variability separate. A run seed combines data-split and optimization randomness.

| Condition | Dataset | Accuracy (%) | Between-target SD | Within-target run-seed SD (RMS) |
|---|---|---|---|---|
| Standard ERM | PACS | 85.88 | 7.98 | 3.04 |
| Standard ERM | VLCS | 77.19 | 14.09 | 1.49 |
| Standard ERM | OfficeHome | 70.21 | 10.90 | 1.79 |
| Standard ERM | TerraIncognita | 47.13 | 6.62 | 4.33 |
| Matched control | PACS | 84.22 | 7.99 | 3.39 |
| Matched control | VLCS | 75.38 | 15.07 | 2.44 |
| Matched control | OfficeHome | 70.05 | 10.73 | 2.31 |
| Matched control | TerraIncognita | 44.14 | 8.70 | 4.21 |
| Full per-domain | PACS | 84.87 | 8.19 | 1.89 |
| Full per-domain | VLCS | 76.42 | 14.90 | 1.65 |
| Full per-domain | OfficeHome | 68.52 | 10.33 | 2.54 |
| Full per-domain | TerraIncognita | 44.17 | 4.60 | 4.28 |
| Frozen | PACS | 86.14 | 7.22 | 2.19 |
| Frozen | VLCS | 75.84 | 14.30 | 2.22 |
| Frozen | OfficeHome | 67.81 | 11.91 | 1.57 |
| Frozen | TerraIncognita | 44.20 | 6.98 | 3.41 |
| No-variance per-domain | PACS | 85.28 | 7.76 | 2.00 |
| No-variance per-domain | VLCS | 76.17 | 14.95 | 1.19 |
| No-variance per-domain | OfficeHome | 68.36 | 10.10 | 2.64 |
| No-variance per-domain | TerraIncognita | 43.70 | 7.90 | 4.72 |
| Shared learnable | PACS | 85.55 | 7.54 | 3.01 |
| Shared learnable | VLCS | 76.56 | 14.65 | 2.20 |
| Shared learnable | OfficeHome | 68.17 | 11.15 | 2.09 |
| Shared learnable | TerraIncognita | 43.75 | 5.35 | 4.73 |

## Primary suite contrasts and per-dataset sensitivity

The two suite tests are the primary Holm family. Their exact p-values sign-flip four dataset-mean effects. Per-dataset exact p-values sign-flip four target effects; their Holm-8 family is a sensitivity, not a second primary family. Evidence against zero and practical equivalence are separate conclusions and may both hold.

| Contrast | Scope | Multiplicity role | Effect (pp) | 95% CI | Exact p | Holm p | Zero-effect conclusion | 90% CI | TOST p | SESOI conclusion |
|---|---|---|---|---|---|---|---|---|---|---|
| Full per-domain - Standard ERM | Suite macro | Primary Holm-2 | -1.61 | [-3.17, -0.04] | 0.125 | 0.250 | No adjusted evidence against zero | [-2.77, -0.45] | 0.848 | Suite macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | Suite macro | Primary Holm-2 | -0.13 | [-0.53, 0.27] | 0.375 | 0.375 | No adjusted evidence against zero | [-0.43, 0.17] | 0.003 | Suite macro equivalent within +/-1.0 pp |
| Full per-domain - Standard ERM | PACS | Sensitivity Holm-8 | -1.01 | [-2.88, 0.86] | 0.250 | 1.000 | No adjusted evidence against zero | [-2.39, 0.37] | 0.506 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | VLCS | Sensitivity Holm-8 | -0.77 | [-3.92, 2.39] | 0.500 | 1.000 | No adjusted evidence against zero | [-3.10, 1.57] | 0.415 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | OfficeHome | Sensitivity Holm-8 | -1.69 | [-4.21, 0.82] | 0.250 | 1.000 | No adjusted evidence against zero | [-3.55, 0.17] | 0.778 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | TerraIncognita | Sensitivity Holm-8 | -2.96 | [-9.18, 3.26] | 0.375 | 1.000 | No adjusted evidence against zero | [-7.56, 1.64] | 0.805 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | PACS | Sensitivity Holm-8 | -0.27 | [-1.47, 0.94] | 0.500 | 1.000 | No adjusted evidence against zero | [-1.16, 0.63] | 0.074 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | VLCS | Sensitivity Holm-8 | -0.39 | [-1.46, 0.68] | 0.375 | 1.000 | No adjusted evidence against zero | [-1.18, 0.40] | 0.084 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | OfficeHome | Sensitivity Holm-8 | 0.19 | [-1.68, 2.06] | 0.875 | 1.000 | No adjusted evidence against zero | [-1.20, 1.57] | 0.130 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | TerraIncognita | Sensitivity Holm-8 | -0.05 | [-4.49, 4.39] | 0.875 | 1.000 | No adjusted evidence against zero | [-3.33, 3.23] | 0.273 | Dataset macro equivalence not demonstrated |

## Source-selected versus final-step sensitivity

| Condition | Dataset | Source-selected (%) | Final-step (%) | Selected - final (pp) |
|---|---|---|---|---|
| Standard ERM | PACS | 85.88 | 85.28 | 0.60 |
| Standard ERM | VLCS | 77.19 | 75.72 | 1.46 |
| Standard ERM | OfficeHome | 70.21 | 67.47 | 2.75 |
| Standard ERM | TerraIncognita | 47.13 | 46.57 | 0.56 |
| Matched control | PACS | 84.22 | 84.04 | 0.17 |
| Matched control | VLCS | 75.38 | 75.63 | -0.25 |
| Matched control | OfficeHome | 70.05 | 67.05 | 3.00 |
| Matched control | TerraIncognita | 44.14 | 43.90 | 0.24 |
| Full per-domain | PACS | 84.87 | 84.50 | 0.38 |
| Full per-domain | VLCS | 76.42 | 75.47 | 0.95 |
| Full per-domain | OfficeHome | 68.52 | 66.03 | 2.49 |
| Full per-domain | TerraIncognita | 44.17 | 43.17 | 1.00 |
| Frozen | PACS | 86.14 | 84.67 | 1.47 |
| Frozen | VLCS | 75.84 | 76.27 | -0.43 |
| Frozen | OfficeHome | 67.81 | 66.13 | 1.68 |
| Frozen | TerraIncognita | 44.20 | 44.30 | -0.10 |
| No-variance per-domain | PACS | 85.28 | 84.57 | 0.71 |
| No-variance per-domain | VLCS | 76.17 | 75.04 | 1.13 |
| No-variance per-domain | OfficeHome | 68.36 | 66.03 | 2.33 |
| No-variance per-domain | TerraIncognita | 43.70 | 43.90 | -0.20 |
| Shared learnable | PACS | 85.55 | 85.10 | 0.45 |
| Shared learnable | VLCS | 76.56 | 76.12 | 0.44 |
| Shared learnable | OfficeHome | 68.17 | 66.27 | 1.90 |
| Shared learnable | TerraIncognita | 43.75 | 43.32 | 0.43 |

## Named-target source-selected accuracies

Every cell reports all four run-seed accuracies plus their mean and run-seed SD. Run seeds vary both the data split and optimization.

| Condition | Dataset | Target | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean | Run-seed SD |
|---|---|---|---|---|---|---|---|---|
| Standard ERM | PACS | ArtPainting | 83.62 | 91.93 | 81.17 | 86.80 | 85.88 | 4.64 |
| Standard ERM | PACS | Cartoon | 82.69 | 81.62 | 79.70 | 75.43 | 79.86 | 3.20 |
| Standard ERM | PACS | Photo | 99.40 | 97.01 | 97.01 | 95.21 | 97.16 | 1.72 |
| Standard ERM | PACS | Sketch | 82.17 | 78.73 | 81.27 | 80.38 | 80.64 | 1.47 |
| Standard ERM | VLCS | Caltech101 | 98.59 | 96.11 | 95.76 | 97.17 | 96.91 | 1.27 |
| Standard ERM | VLCS | LabelMe | 66.42 | 63.21 | 64.53 | 62.83 | 64.25 | 1.62 |
| Standard ERM | VLCS | SUN09 | 71.19 | 69.36 | 71.34 | 71.80 | 70.92 | 1.07 |
| Standard ERM | VLCS | VOC2007 | 78.52 | 75.70 | 77.93 | 74.52 | 76.67 | 1.88 |
| Standard ERM | OfficeHome | Art | 67.01 | 64.54 | 64.54 | 66.39 | 65.62 | 1.28 |
| Standard ERM | OfficeHome | Clipart | 57.16 | 53.15 | 58.53 | 59.11 | 56.99 | 2.69 |
| Standard ERM | OfficeHome | Product | 79.93 | 78.58 | 77.00 | 77.11 | 78.16 | 1.38 |
| Standard ERM | OfficeHome | RealWorld | 80.25 | 78.19 | 80.25 | 81.63 | 80.08 | 1.42 |
| Standard ERM | TerraIncognita | L100 | 45.85 | 47.38 | 46.15 | 57.33 | 49.18 | 5.48 |
| Standard ERM | TerraIncognita | L38 | 48.08 | 49.31 | 42.45 | 36.05 | 43.97 | 6.07 |
| Standard ERM | TerraIncognita | L43 | 56.09 | 54.98 | 53.11 | 57.09 | 55.32 | 1.71 |
| Standard ERM | TerraIncognita | L46 | 42.32 | 40.20 | 36.93 | 40.77 | 40.05 | 2.27 |
| Matched control | PACS | ArtPainting | 86.06 | 86.55 | 82.15 | 83.86 | 84.66 | 2.04 |
| Matched control | PACS | Cartoon | 81.84 | 79.70 | 77.14 | 73.93 | 78.15 | 3.41 |
| Matched control | PACS | Photo | 96.11 | 93.71 | 95.81 | 95.81 | 95.36 | 1.11 |
| Matched control | PACS | Sketch | 83.18 | 82.93 | 71.97 | 76.69 | 78.69 | 5.39 |
| Matched control | VLCS | Caltech101 | 95.41 | 96.11 | 96.11 | 98.23 | 96.47 | 1.22 |
| Matched control | VLCS | LabelMe | 60.75 | 61.13 | 59.62 | 61.32 | 60.71 | 0.76 |
| Matched control | VLCS | SUN09 | 74.09 | 70.58 | 72.87 | 72.87 | 72.60 | 1.46 |
| Matched control | VLCS | VOC2007 | 74.96 | 75.41 | 70.67 | 65.93 | 71.74 | 4.43 |
| Matched control | OfficeHome | Art | 64.12 | 65.15 | 62.27 | 69.69 | 65.31 | 3.16 |
| Matched control | OfficeHome | Clipart | 59.11 | 53.61 | 56.47 | 59.34 | 57.13 | 2.68 |
| Matched control | OfficeHome | Product | 78.80 | 80.27 | 78.92 | 76.44 | 78.61 | 1.59 |
| Matched control | OfficeHome | RealWorld | 78.65 | 77.61 | 80.02 | 80.37 | 79.16 | 1.27 |
| Matched control | TerraIncognita | L100 | 58.46 | 49.23 | 56.51 | 57.64 | 55.46 | 4.23 |
| Matched control | TerraIncognita | L38 | 42.55 | 47.31 | 42.45 | 33.44 | 41.44 | 5.80 |
| Matched control | TerraIncognita | L43 | 42.66 | 50.37 | 43.53 | 43.66 | 45.06 | 3.57 |
| Matched control | TerraIncognita | L46 | 38.07 | 35.13 | 32.19 | 33.09 | 34.62 | 2.61 |
| Full per-domain | PACS | ArtPainting | 84.11 | 85.33 | 83.62 | 83.86 | 84.23 | 0.76 |
| Full per-domain | PACS | Cartoon | 78.85 | 78.85 | 76.28 | 76.71 | 77.67 | 1.37 |
| Full per-domain | PACS | Photo | 95.51 | 97.31 | 96.71 | 96.41 | 96.48 | 0.75 |
| Full per-domain | PACS | Sketch | 81.40 | 85.61 | 79.75 | 77.71 | 81.11 | 3.35 |
| Full per-domain | VLCS | Caltech101 | 99.29 | 96.47 | 97.17 | 96.82 | 97.44 | 1.27 |
| Full per-domain | VLCS | LabelMe | 62.08 | 62.26 | 62.83 | 62.08 | 62.31 | 0.36 |
| Full per-domain | VLCS | SUN09 | 73.32 | 69.82 | 73.93 | 71.65 | 72.18 | 1.85 |
| Full per-domain | VLCS | VOC2007 | 75.70 | 75.41 | 73.33 | 70.52 | 73.74 | 2.39 |
| Full per-domain | OfficeHome | Art | 65.36 | 62.27 | 56.91 | 65.36 | 62.47 | 3.99 |
| Full per-domain | OfficeHome | Clipart | 55.90 | 55.21 | 57.96 | 59.22 | 57.07 | 1.85 |
| Full per-domain | OfficeHome | Product | 79.26 | 79.14 | 74.75 | 76.21 | 77.34 | 2.23 |
| Full per-domain | OfficeHome | RealWorld | 76.46 | 76.00 | 78.87 | 77.38 | 77.18 | 1.27 |
| Full per-domain | TerraIncognita | L100 | 48.00 | 44.82 | 55.28 | 48.31 | 49.10 | 4.41 |
| Full per-domain | TerraIncognita | L38 | 34.10 | 44.91 | 43.16 | 39.27 | 40.36 | 4.79 |
| Full per-domain | TerraIncognita | L43 | 40.42 | 49.50 | 48.88 | 49.50 | 47.08 | 4.45 |
| Full per-domain | TerraIncognita | L46 | 42.65 | 35.62 | 39.62 | 42.65 | 40.13 | 3.33 |
| Frozen | PACS | ArtPainting | 88.26 | 88.51 | 84.11 | 83.86 | 86.19 | 2.54 |
| Frozen | PACS | Cartoon | 84.19 | 77.78 | 77.35 | 79.06 | 79.59 | 3.15 |
| Frozen | PACS | Photo | 95.81 | 95.51 | 96.71 | 96.71 | 96.18 | 0.62 |
| Frozen | PACS | Sketch | 82.17 | 84.59 | 82.68 | 80.89 | 82.58 | 1.53 |
| Frozen | VLCS | Caltech101 | 98.23 | 94.70 | 96.82 | 95.05 | 96.20 | 1.64 |
| Frozen | VLCS | LabelMe | 62.83 | 65.66 | 61.13 | 61.32 | 62.74 | 2.09 |
| Frozen | VLCS | SUN09 | 74.54 | 68.45 | 74.70 | 73.32 | 72.75 | 2.94 |
| Frozen | VLCS | VOC2007 | 70.37 | 74.22 | 72.30 | 69.78 | 71.67 | 2.01 |
| Frozen | OfficeHome | Art | 62.06 | 61.03 | 60.82 | 63.30 | 61.80 | 1.13 |
| Frozen | OfficeHome | Clipart | 53.95 | 52.00 | 55.33 | 54.64 | 53.98 | 1.43 |
| Frozen | OfficeHome | Product | 78.92 | 75.99 | 77.00 | 76.44 | 77.09 | 1.29 |
| Frozen | OfficeHome | RealWorld | 76.23 | 76.81 | 79.79 | 80.71 | 78.39 | 2.20 |
| Frozen | TerraIncognita | L100 | 55.18 | 46.67 | 49.33 | 57.23 | 52.10 | 4.93 |
| Frozen | TerraIncognita | L38 | 37.79 | 42.75 | 40.91 | 37.79 | 39.81 | 2.45 |
| Frozen | TerraIncognita | L43 | 43.66 | 45.77 | 50.00 | 51.99 | 47.85 | 3.82 |
| Frozen | TerraIncognita | L46 | 36.36 | 36.52 | 36.44 | 38.89 | 37.05 | 1.23 |
| No-variance per-domain | PACS | ArtPainting | 83.62 | 82.89 | 87.53 | 82.89 | 84.23 | 2.23 |
| No-variance per-domain | PACS | Cartoon | 79.27 | 77.56 | 77.56 | 77.99 | 78.10 | 0.81 |
| No-variance per-domain | PACS | Photo | 97.60 | 94.61 | 95.81 | 97.01 | 96.26 | 1.33 |
| No-variance per-domain | PACS | Sketch | 84.71 | 85.22 | 79.11 | 81.15 | 82.55 | 2.92 |
| No-variance per-domain | VLCS | Caltech101 | 97.88 | 97.17 | 96.82 | 98.23 | 97.53 | 0.65 |
| No-variance per-domain | VLCS | LabelMe | 63.02 | 63.21 | 63.96 | 60.38 | 62.64 | 1.56 |
| No-variance per-domain | VLCS | SUN09 | 72.26 | 71.04 | 72.56 | 70.73 | 71.65 | 0.90 |
| No-variance per-domain | VLCS | VOC2007 | 73.63 | 73.93 | 73.04 | 70.81 | 72.85 | 1.41 |
| No-variance per-domain | OfficeHome | Art | 60.62 | 59.38 | 68.25 | 60.41 | 62.16 | 4.09 |
| No-variance per-domain | OfficeHome | Clipart | 57.27 | 54.07 | 56.82 | 61.40 | 57.39 | 3.02 |
| No-variance per-domain | OfficeHome | Product | 76.89 | 78.69 | 75.99 | 75.99 | 76.89 | 1.28 |
| No-variance per-domain | OfficeHome | RealWorld | 77.04 | 76.35 | 76.92 | 77.61 | 76.98 | 0.52 |
| No-variance per-domain | TerraIncognita | L100 | 60.10 | 52.41 | 51.59 | 46.77 | 52.72 | 5.52 |
| No-variance per-domain | TerraIncognita | L38 | 39.22 | 35.28 | 44.55 | 33.08 | 38.03 | 5.03 |
| No-variance per-domain | TerraIncognita | L43 | 49.25 | 43.66 | 47.39 | 51.12 | 47.85 | 3.19 |
| No-variance per-domain | TerraIncognita | L46 | 43.06 | 32.68 | 35.95 | 33.09 | 36.19 | 4.80 |
| Shared learnable | PACS | ArtPainting | 80.68 | 88.26 | 84.84 | 87.04 | 85.21 | 3.33 |
| Shared learnable | PACS | Cartoon | 83.33 | 79.91 | 75.00 | 77.35 | 78.90 | 3.57 |
| Shared learnable | PACS | Photo | 97.31 | 97.31 | 95.81 | 94.31 | 96.18 | 1.43 |
| Shared learnable | PACS | Sketch | 84.08 | 84.84 | 77.83 | 80.89 | 81.91 | 3.21 |
| Shared learnable | VLCS | Caltech101 | 97.88 | 97.17 | 96.11 | 97.88 | 97.26 | 0.83 |
| Shared learnable | VLCS | LabelMe | 65.85 | 59.81 | 64.15 | 61.51 | 62.83 | 2.69 |
| Shared learnable | VLCS | SUN09 | 73.02 | 68.75 | 74.39 | 71.65 | 71.95 | 2.41 |
| Shared learnable | VLCS | VOC2007 | 76.59 | 75.85 | 72.30 | 72.00 | 74.19 | 2.37 |
| Shared learnable | OfficeHome | Art | 61.03 | 59.18 | 63.09 | 64.95 | 62.06 | 2.50 |
| Shared learnable | OfficeHome | Clipart | 57.04 | 52.46 | 53.84 | 58.76 | 55.53 | 2.89 |
| Shared learnable | OfficeHome | Product | 78.58 | 79.26 | 76.10 | 76.78 | 77.68 | 1.48 |
| Shared learnable | OfficeHome | RealWorld | 76.46 | 78.19 | 77.15 | 77.84 | 77.41 | 0.76 |
| Shared learnable | TerraIncognita | L100 | 57.54 | 46.77 | 49.23 | 45.03 | 49.64 | 5.54 |
| Shared learnable | TerraIncognita | L38 | 44.75 | 47.52 | 38.76 | 33.69 | 41.18 | 6.19 |
| Shared learnable | TerraIncognita | L43 | 45.15 | 41.79 | 50.50 | 48.63 | 46.52 | 3.85 |
| Shared learnable | TerraIncognita | L46 | 40.77 | 34.97 | 37.99 | 36.93 | 37.66 | 2.42 |

## Named-target effects for the primary contrasts

| Contrast | Dataset | Target | Mean paired effect (pp) | Run-seed-pair SD |
|---|---|---|---|---|
| Full per-domain - Standard ERM | PACS | ArtPainting | -1.65 | 3.98 |
| Full per-domain - Standard ERM | PACS | Cartoon | -2.19 | 2.36 |
| Full per-domain - Standard ERM | PACS | Photo | -0.67 | 2.23 |
| Full per-domain - Standard ERM | PACS | Sketch | 0.48 | 4.34 |
| Full per-domain - Standard ERM | VLCS | Caltech101 | 0.53 | 0.74 |
| Full per-domain - Standard ERM | VLCS | LabelMe | -1.93 | 1.65 |
| Full per-domain - Standard ERM | VLCS | SUN09 | 1.26 | 1.31 |
| Full per-domain - Standard ERM | VLCS | VOC2007 | -2.93 | 1.90 |
| Full per-domain - Standard ERM | OfficeHome | Art | -3.14 | 3.03 |
| Full per-domain - Standard ERM | OfficeHome | Clipart | 0.09 | 1.43 |
| Full per-domain - Standard ERM | OfficeHome | Product | -0.82 | 1.15 |
| Full per-domain - Standard ERM | OfficeHome | RealWorld | -2.90 | 1.35 |
| Full per-domain - Standard ERM | TerraIncognita | L100 | -0.08 | 7.66 |
| Full per-domain - Standard ERM | TerraIncognita | L38 | -3.61 | 7.61 |
| Full per-domain - Standard ERM | TerraIncognita | L43 | -8.24 | 5.14 |
| Full per-domain - Standard ERM | TerraIncognita | L46 | 0.08 | 3.26 |
| No-variance per-domain - Shared learnable | PACS | ArtPainting | -0.98 | 4.41 |
| No-variance per-domain - Shared learnable | PACS | Cartoon | -0.80 | 2.97 |
| No-variance per-domain - Shared learnable | PACS | Photo | 0.07 | 2.21 |
| No-variance per-domain - Shared learnable | PACS | Sketch | 0.64 | 0.45 |
| No-variance per-domain - Shared learnable | VLCS | Caltech101 | 0.27 | 0.34 |
| No-variance per-domain - Shared learnable | VLCS | LabelMe | -0.19 | 2.63 |
| No-variance per-domain - Shared learnable | VLCS | SUN09 | -0.30 | 1.79 |
| No-variance per-domain - Shared learnable | VLCS | VOC2007 | -1.33 | 1.56 |
| No-variance per-domain - Shared learnable | OfficeHome | Art | 0.10 | 3.97 |
| No-variance per-domain - Shared learnable | OfficeHome | Clipart | 1.86 | 1.24 |
| No-variance per-domain - Shared learnable | OfficeHome | Product | -0.79 | 0.66 |
| No-variance per-domain - Shared learnable | OfficeHome | RealWorld | -0.43 | 1.01 |
| No-variance per-domain - Shared learnable | TerraIncognita | L100 | 3.08 | 1.74 |
| No-variance per-domain - Shared learnable | TerraIncognita | L38 | -3.15 | 7.63 |
| No-variance per-domain - Shared learnable | TerraIncognita | L43 | 1.34 | 3.11 |
| No-variance per-domain - Shared learnable | TerraIncognita | L46 | -1.47 | 2.63 |

## Paired-effect variability decomposition

These descriptive fields keep target, run-seed, and interaction variability separate for each primary paired effect. They are not additional inferential tests.

| Contrast | Dataset | Mean effect (pp) | Between-target SD | Between-run-seed SD | Within-target run-seed SD (RMS) | Target x run-seed residual SD |
|---|---|---|---|---|---|---|
| Full per-domain - Standard ERM | PACS | -1.01 | 1.17 | 0.67 | 3.36 | 3.80 |
| Full per-domain - Standard ERM | VLCS | -0.77 | 1.98 | 0.54 | 1.47 | 1.58 |
| Full per-domain - Standard ERM | OfficeHome | -1.69 | 1.58 | 1.03 | 1.90 | 1.84 |
| Full per-domain - Standard ERM | TerraIncognita | -2.96 | 3.91 | 3.73 | 6.20 | 5.71 |
| No-variance per-domain - Shared learnable | PACS | -0.27 | 0.76 | 1.70 | 2.88 | 2.69 |
| No-variance per-domain - Shared learnable | VLCS | -0.39 | 0.67 | 1.08 | 1.78 | 1.63 |
| No-variance per-domain - Shared learnable | OfficeHome | 0.19 | 1.18 | 1.20 | 2.17 | 2.08 |
| No-variance per-domain - Shared learnable | TerraIncognita | -0.05 | 2.79 | 1.21 | 4.41 | 4.90 |

## Held-out target in-split sensitivity

The target out-split remains primary. These source-selected target in-split contrasts are sensitivity results and never enter checkpoint selection.

| Contrast | Dataset | Target-in effect (pp) | 95% CI | Exact p |
|---|---|---|---|---|
| Full per-domain - Standard ERM | PACS | -0.64 | [-2.12, 0.84] | 0.375 |
| Full per-domain - Standard ERM | VLCS | -0.98 | [-3.88, 1.92] | 0.375 |
| Full per-domain - Standard ERM | OfficeHome | -1.56 | [-3.84, 0.72] | 0.250 |
| Full per-domain - Standard ERM | TerraIncognita | -2.49 | [-7.33, 2.35] | 0.250 |
| No-variance per-domain - Shared learnable | PACS | -0.25 | [-3.09, 2.59] | 0.875 |
| No-variance per-domain - Shared learnable | VLCS | -0.51 | [-1.68, 0.66] | 0.250 |
| No-variance per-domain - Shared learnable | OfficeHome | 0.17 | [-1.52, 1.85] | 1.000 |
| No-variance per-domain - Shared learnable | TerraIncognita | 0.53 | [-3.83, 4.89] | 0.875 |

## SESOI and equivalence sensitivity

Equivalence is reported only when the analyzer's relevant clustered 90% interval lies inside the stated symmetric margin and its TOST p-value is below 0.05. The +/-1.0 pp SESOI was frozen at the revision stage; it is not represented as preregistered or prespecified. Margins of +/-0.5 and +/-2.0 pp are sensitivity analyses. A suite conclusion concerns the equal-dataset macro effect, not every target.

| Contrast | Scope | Analysis role | Margin (+/- pp) | Effect (pp) | 90% CI | TOST p | Macro-equivalence conclusion |
|---|---|---|---|---|---|---|---|
| Full per-domain - Standard ERM | Suite macro | Sensitivity | 0.5 | -1.61 | [-2.77, -0.45] | 0.945 | Suite macro equivalence not demonstrated |
| Full per-domain - Standard ERM | Suite macro | Sensitivity | 1.0 | -1.61 | [-2.77, -0.45] | 0.848 | Suite macro equivalence not demonstrated |
| Full per-domain - Standard ERM | Suite macro | Sensitivity | 2.0 | -1.61 | [-2.77, -0.45] | 0.242 | Suite macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | Suite macro | Sensitivity | 0.5 | -0.13 | [-0.43, 0.17] | 0.031 | Suite macro equivalent within +/-0.5 pp |
| No-variance per-domain - Shared learnable | Suite macro | Primary revision-stage equivalence | 1.0 | -0.13 | [-0.43, 0.17] | 0.003 | Suite macro equivalent within +/-1.0 pp |
| No-variance per-domain - Shared learnable | Suite macro | Sensitivity | 2.0 | -0.13 | [-0.43, 0.17] | <0.001 | Suite macro equivalent within +/-2.0 pp |
| Full per-domain - Standard ERM | PACS | Sensitivity | 0.5 | -1.01 | [-2.39, 0.37] | 0.775 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | PACS | Sensitivity | 1.0 | -1.01 | [-2.39, 0.37] | 0.506 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | PACS | Sensitivity | 2.0 | -1.01 | [-2.39, 0.37] | 0.095 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | VLCS | Sensitivity | 0.5 | -0.77 | [-3.10, 1.57] | 0.598 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | VLCS | Sensitivity | 1.0 | -0.77 | [-3.10, 1.57] | 0.415 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | VLCS | Sensitivity | 2.0 | -0.77 | [-3.10, 1.57] | 0.151 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | OfficeHome | Sensitivity | 0.5 | -1.69 | [-3.55, 0.17] | 0.886 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | OfficeHome | Sensitivity | 1.0 | -1.69 | [-3.55, 0.17] | 0.778 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | OfficeHome | Sensitivity | 2.0 | -1.69 | [-3.55, 0.17] | 0.362 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | TerraIncognita | Sensitivity | 0.5 | -2.96 | [-7.56, 1.64] | 0.851 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | TerraIncognita | Sensitivity | 1.0 | -2.96 | [-7.56, 1.64] | 0.805 | Dataset macro equivalence not demonstrated |
| Full per-domain - Standard ERM | TerraIncognita | Sensitivity | 2.0 | -2.96 | [-7.56, 1.64] | 0.672 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | PACS | Sensitivity | 0.5 | -0.27 | [-1.16, 0.63] | 0.291 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | PACS | Sensitivity | 1.0 | -0.27 | [-1.16, 0.63] | 0.074 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | PACS | Sensitivity | 2.0 | -0.27 | [-1.16, 0.63] | 0.010 | Dataset macro equivalent within +/-2.0 pp |
| No-variance per-domain - Shared learnable | VLCS | Sensitivity | 0.5 | -0.39 | [-1.18, 0.40] | 0.383 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | VLCS | Sensitivity | 1.0 | -0.39 | [-1.18, 0.40] | 0.084 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | VLCS | Sensitivity | 2.0 | -0.39 | [-1.18, 0.40] | 0.009 | Dataset macro equivalent within +/-2.0 pp |
| No-variance per-domain - Shared learnable | OfficeHome | Sensitivity | 0.5 | 0.19 | [-1.20, 1.57] | 0.315 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | OfficeHome | Sensitivity | 1.0 | 0.19 | [-1.20, 1.57] | 0.130 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | OfficeHome | Sensitivity | 2.0 | 0.19 | [-1.20, 1.57] | 0.027 | Dataset macro equivalent within +/-2.0 pp |
| No-variance per-domain - Shared learnable | TerraIncognita | Sensitivity | 0.5 | -0.05 | [-3.33, 3.23] | 0.384 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | TerraIncognita | Sensitivity | 1.0 | -0.05 | [-3.33, 3.23] | 0.273 | Dataset macro equivalence not demonstrated |
| No-variance per-domain - Shared learnable | TerraIncognita | Sensitivity | 2.0 | -0.05 | [-3.33, 3.23] | 0.128 | Dataset macro equivalence not demonstrated |
