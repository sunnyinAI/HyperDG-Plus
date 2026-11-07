# Reviewer-round analysis status

- Complete runs available: **384/384** (six conditions x four datasets x four targets x four seeds).
- Missing or incomplete runs: **0**.
- Primary checkpoint rule: maximum mean source-domain out-split validation accuracy; the target domain is excluded; ties use the earliest checkpoint.
- Inferential unit: held-out target domain after averaging the four paired run-seed differences within each benchmark; the suite-level unit is the benchmark dataset.
- Exact two-sided sign-flip tests have only 16 assignments with four target domains or four benchmark datasets, so their smallest attainable p-value is 0.125.
- Frozen revision-stage SESOI: +/-1.0 percentage point.
- A run seed changes the 80/20 data split as well as initialization, minibatch order, and augmentation.

## Source-validation-selected condition summaries

| condition | dataset | n_domains | macro_mean_accuracy | between_target_sd_of_means | rms_within_target_run_seed_sd |
|---|---|---|---|---|---|
| standard_erm | pacs | 4 | 85.883 | 7.976 | 3.040 |
| standard_erm | vlcs | 4 | 77.186 | 14.094 | 1.492 |
| standard_erm | officehome | 4 | 70.211 | 10.901 | 1.787 |
| standard_erm | terraincognita | 4 | 47.130 | 6.615 | 4.327 |
| matched_control | pacs | 4 | 84.216 | 7.992 | 3.394 |
| matched_control | vlcs | 4 | 75.378 | 15.065 | 2.440 |
| matched_control | officehome | 4 | 70.052 | 10.734 | 2.309 |
| matched_control | terraincognita | 4 | 44.143 | 8.698 | 4.215 |
| full_per_domain | pacs | 4 | 84.874 | 8.189 | 1.888 |
| full_per_domain | vlcs | 4 | 76.418 | 14.899 | 1.649 |
| full_per_domain | officehome | 4 | 68.517 | 10.334 | 2.544 |
| full_per_domain | terraincognita | 4 | 44.169 | 4.603 | 4.280 |
| frozen | pacs | 4 | 86.136 | 7.220 | 2.186 |
| frozen | vlcs | 4 | 75.839 | 14.298 | 2.222 |
| frozen | officehome | 4 | 67.814 | 11.906 | 1.568 |
| frozen | terraincognita | 4 | 44.205 | 6.981 | 3.406 |
| no_variance | pacs | 4 | 85.283 | 7.760 | 1.995 |
| no_variance | vlcs | 4 | 76.167 | 14.951 | 1.188 |
| no_variance | officehome | 4 | 68.356 | 10.096 | 2.635 |
| no_variance | terraincognita | 4 | 43.699 | 7.897 | 4.716 |
| shared_learnable | pacs | 4 | 85.550 | 7.542 | 3.010 |
| shared_learnable | vlcs | 4 | 76.557 | 14.651 | 2.201 |
| shared_learnable | officehome | 4 | 68.169 | 11.150 | 2.085 |
| shared_learnable | terraincognita | 4 | 43.751 | 5.355 | 4.735 |

## Primary suite-level reviewer-requested contrasts

| contrast | effect_pp | ci95_t_low | ci95_t_high | exact_4_dataset_signflip_p | holm_p_primary_2 | ci90_t_low | ci90_t_high | equivalent_at_0p05 |
|---|---|---|---|---|---|---|---|---|
| full_minus_standard_erm | -1.608 | -3.173 | -0.043 | 0.125 | 0.250 | -2.766 | -0.451 | False |
| no_variance_minus_shared | -0.131 | -0.534 | 0.273 | 0.375 | 0.375 | -0.429 | 0.168 | True |

## Per-dataset estimates and Holm-8 sensitivity

| contrast | dataset | effect_pp | ci95_t_low | ci95_t_high | exact_signflip_p | holm_p_dataset_8_sensitivity | ci90_t_low | ci90_t_high | equivalent_at_0p05 |
|---|---|---|---|---|---|---|---|---|---|
| full_minus_standard_erm | pacs | -1.009 | -2.876 | 0.858 | 0.250 | 1.000 | -2.390 | 0.371 | False |
| full_minus_standard_erm | vlcs | -0.768 | -3.924 | 2.388 | 0.500 | 1.000 | -3.102 | 1.566 | False |
| full_minus_standard_erm | officehome | -1.694 | -4.208 | 0.821 | 0.250 | 1.000 | -3.553 | 0.166 | False |
| full_minus_standard_erm | terraincognita | -2.961 | -9.183 | 3.260 | 0.375 | 1.000 | -7.562 | 1.640 | False |
| no_variance_minus_shared | pacs | -0.267 | -1.473 | 0.940 | 0.500 | 1.000 | -1.159 | 0.625 | False |
| no_variance_minus_shared | vlcs | -0.390 | -1.464 | 0.684 | 0.375 | 1.000 | -1.185 | 0.404 | False |
| no_variance_minus_shared | officehome | 0.186 | -1.684 | 2.057 | 0.875 | 1.000 | -1.197 | 1.569 | False |
| no_variance_minus_shared | terraincognita | -0.051 | -4.488 | 4.385 | 0.875 | 1.000 | -3.332 | 3.230 | False |

## Interpretation guardrails

- A non-significant sign-flip test is not evidence of equivalence.
- The primary equivalence question is the source-selected suite NoVariance-versus-SharedLearnable macro at the revision-stage margin. Dataset and alternate-margin TOSTs are sensitivities.
- Macro equivalence concerns the equal-dataset/equal-target average, not every named target.
- Target learning curves are post-hoc diagnostics and never enter checkpoint selection.
- The clean domain-specificity contrast is NoVariance per-domain minus SharedLearnable, because the variance term is structurally zero for one shared scalar.
