# HyperDG2 implementation and evidence audit

**Audit snapshot:** 23 July 2026, after excluding all legacy shared-curvature
reuse and generating the clean 128-run added-condition grid.

This document pins the implementation, data path, condition identities, and
evidence boundaries for the Paper 10466 reviewer round. It is an audit of the
executed scalar curvature-conditioned feature-scaling code. It is not evidence
for a Lorentz classifier, general hyperbolic representation learning, or
Sharpness-Aware Minimization (SAM).

## 1. Evidence ledger

The submitted primary grid contained

```text
4 conditions x 4 datasets x 4 held-out targets x 4 seeds = 256 runs.
```

All 256 legacy primary cells are complete under the current inclusion rule:
`metrics.jsonl` contains the exact 25-record schedule from step 200 through
4800 plus 4999, every record contains all environment `in`/`out` accuracy
keys, and `model_last.pt` exists.

| Analysis condition | Expected | Final step 4999 | Checkpoint present |
|---|---:|---:|---:|
| `matched_control` | 64 | 64 | 64 |
| `full_per_domain` | 64 | 64 | 64 |
| `frozen` | 64 | 64 | 64 |
| `no_variance` | 64 | 64 | 64 |
| **Legacy primary total** | **256** | **256** | **256** |

The revised six-condition grid contains 384 cells. At this snapshot,
`outputs/reviewer_round2/readiness.json` reports 256 complete evidence cells:
the four 64-run legacy conditions above. No legacy SharedLearnable run is
admitted to the confirmatory matrix.

| Added condition | Planned cells | Reused legacy cells | New complete cells at frozen snapshot | Clean runs required |
|---|---:|---:|---:|---:|
| `standard_erm` | 64 | 0 | 0 | 64 |
| `shared_learnable` | 64 | 0 | 0 | 64 |
| **Added-condition total** | **128** | **0** | **0** | **128** |

The 128-run burden is therefore exact:

- all 64 StandardERM cells;
- all 64 SharedLearnable cells.

The four generated GPU queues contain 32 configurations each. Fifteen
complete-looking legacy SharedLearnable cells remain at
`experiments/reviewer_hQnG/single_curv/{dataset}_split{target}`, but they are
excluded: twelve PACS/VLCS/OfficeHome cells were run on 30 March 2026 with
Torch/Torchvision resolving from the conda environment (Torch 2.2.2 and
Torchvision 0.17.2), while the 256 primary comparators and current jobs resolve
user-site Torch 2.6.0 and Torchvision 0.21.0. Three completed TerraIncognita
cells were produced around that environment transition, and the fourth stops
at step 2600 without a checkpoint. Mixing these cells would weaken the matched
comparison.

Legacy component-removal, variance-sweep, old SharedLearnable, and other
diagnostic artifacts are outside the 384-cell confirmatory matrix. No
diagnostic run may be added to the denominator or used to fill a missing
primary cell.

## 2. Exact condition identities

Let the five HyperDG coefficients be:

```text
A = alpha_curv
B = beta_align
G = gamma_sharp
L = delta_local_flat
D = delta_global_flat
```

All HyperDG-path conditions initialize each log-curvature parameter to zero, so
its positive scale starts at
`softplus(0) + 1e-4`, approximately 0.693247. They also instantiate a separate
global log-curvature parameter. “Per-domain” means one scalar for each of the
three source-loader slots in a leave-one-domain-out run, not four simultaneous
dataset-domain scalars.

| Analysis label | Paper-facing label | Algorithm route and curvature | Exact coefficients `(A,B,G,L,D)` | What the contrast actually changes |
|---|---|---|---|---|
| `standard_erm` | StandardERM | DomainBed `ERM`; no curvature parameters or scaling | not applicable | Canonical concatenated-minibatch ERM update |
| `matched_control` | ZeroPenalty | `hyperdg`; three fixed local scalars and one fixed global scalar; `curvature_lr=0` | `(0,0,0,0,0)` | HyperDG code path with auxiliary losses disabled; local training features are still divided by the fixed scale |
| `full_per_domain` | FullPerDomain | `hyperdg`; three learned local scalars and one learned global scalar; `curvature_lr=1e-4` | `(0.2,0.5,0.05,0.1,0.1)` | Full submitted implementation |
| `frozen` | Frozen | `hyperdg`; three fixed local scalars and one fixed global scalar; `curvature_lr=0` | `(0,0.5,0.05,0.1,0.1)` | Retains the non-dispersion auxiliary package while all curvature values remain at initialization |
| `no_variance` | NoVariance | `hyperdg`; three learned local scalars and one learned global scalar; `curvature_lr=1e-4` | `(0,0.5,0.05,0.1,0.1)` | Removes only the explicit pairwise curvature-dispersion coefficient from FullPerDomain |
| `shared_learnable` | SharedLearnable | `hyperdg`; one learned local scalar shared by all three source loaders plus one learned global scalar; `curvature_lr=1e-4` | `(0.2,0.5,0.05,0.1,0.1)` | Replaces the three learned local scalars with one; the curvature-dispersion loss is structurally zero despite `A=0.2` |

The directory mapping used by `scripts/analyze_review_round2.py` is:

| Condition | Seed 0 evidence | Seeds 1--3 evidence |
|---|---|---|
| `matched_control` | `experiments/batch2/erm/{dataset}_split{target}` | `experiments/batch4/seed{seed}/erm/{dataset}_split{target}` |
| `full_per_domain` | `experiments/batch3/baseline/{dataset}_split{target}` | `experiments/batch4/seed{seed}/baseline/{dataset}_split{target}` |
| `frozen` | `experiments/batch1/fixed/frozen/{dataset}_split{target}` | `experiments/batch4/seed{seed}/fixed/{dataset}_split{target}` |
| `no_variance` | `experiments/batch1/novar/a0/{dataset}_split{target}` | `experiments/batch4/seed{seed}/novar/{dataset}_split{target}` |
| `standard_erm` | `HyperDG2/experiments/reviewer_round2/standard_erm/seed0/{dataset}_split{target}` | `HyperDG2/experiments/reviewer_round2/standard_erm/seed{seed}/{dataset}_split{target}` |
| `shared_learnable` | `HyperDG2/experiments/reviewer_round2/shared_learnable/seed0/{dataset}_split{target}` | `HyperDG2/experiments/reviewer_round2/shared_learnable/seed{seed}/{dataset}_split{target}` |

The legacy folder name `erm` must never be reported as canonical ERM. It is the
`matched_control`/ZeroPenalty condition: its YAML says `algorithm: hyperdg`.
Conversely, the new StandardERM YAML says `algorithm: ERM` and is routed to the
unmodified DomainBed ERM class. StandardERM has no curvature optimizer and no
HyperDG gradient clipping.

## 3. Shared protocol and data resolution

All six conditions resolve to the following common settings:

| Setting | Frozen value |
|---|---|
| Seeds | 0, 1, 2, 3 |
| Held-out targets | indices 0, 1, 2, 3, one per run |
| Holdout fraction | 0.20 within every environment |
| Split seed | `seed_hash(seed, environment_index)` |
| Source batch size | 32 per source environment |
| Updates | 5,000, numbered 0--4999 |
| Evaluation | every 200 updates and at 4999; no step-0 evaluation |
| Network learning rate | `5e-5` |
| Weight decay | 0 |
| Class balancing | false |
| Data augmentation | true |
| Classifier | linear, with bias |
| Source-selection score | equal-weight mean of the three source `env*_out_acc` values |
| Source-selection tie | earliest checkpoint |
| Primary target outcome | held-out target `out` accuracy at the source-selected checkpoint |
| Sensitivity outcomes | update-4999 target `out` accuracy and source-selected target `in` accuracy |

The old 256 YAML files omit some DomainBed defaults. At execution they resolve to
`data_augmentation=True`, `resnet18=False`, `resnet50_augmix=True`,
`class_balanced=False`, `freeze_bn=False`, and `resnet_dropout=0`. The new YAML
files state those values explicitly to prevent default drift.

Dataset and target-index resolution is:

| Dataset config | Runtime root | Target 0 | Target 1 | Target 2 | Target 3 |
|---|---|---|---|---|---|
| `configs/datasets/pacs.yaml` | `data/PACS` | Art Painting | Cartoon | Photo | Sketch |
| `configs/datasets/vlcs.yaml` | `data/VLCS` | Caltech101 | LabelMe | SUN09 | VOC2007 |
| `configs/datasets/officehome.yaml` | `data/officehome_clean/office_home` | Art | Clipart | Product | Real World |
| `configs/datasets/terraincognita.yaml` | `data/terra_incognita` | L100 | L38 | L43 | L46 |

Each config fixes `num_workers=8`. The present PACS, VLCS, and OfficeHome
environment folders are symbolic links into other local research-data trees;
TerraIncognita is stored directly in this checkout. These symlink targets are
machine-specific and must be replaced by a documented portable dataset
provisioning step before an archival reproduction claim.

### Augmented source validation

The DomainBed dataset object chooses an environment transform before the
environment is divided into its 80% `in` and 20% `out` subsets. For source
environments, both subsets therefore retain the stochastic source transform:
random resized crop to 224 with scale 0.7--1.0, random horizontal flip, color
jitter, random grayscale, tensor conversion, and ImageNet normalization. The
held-out target uses deterministic resize to 224 and ImageNet normalization.

Consequently, the source `out` scores used for checkpoint selection are
evaluated through stochastic augmentation. This behavior is shared by all
existing runs and is a protocol limitation; it is not a clean deterministic
validation pipeline.

## 4. Actual backbone and optimizer resolution

The active feature extractor is not merely an unspecified “ImageNet-pretrained
ResNet-50.” With `resnet18=False` and `resnet50_augmix=True`,
`third_party/domainbed/domainbed/networks.py` replaces the initially constructed
torchvision model with:

```python
timm.create_model("resnet50.ram_in1k", pretrained=True)
```

The active output is a 2,048-dimensional feature vector followed by the linear
classifier. Batch normalization is not frozen and dropout is zero. The active
interpreter resolves `timm==1.0.24`, Torch 2.6.0+cu124, and Torchvision
0.21.0+cu124; the conda environment without user-site packages instead
contains Torch 2.2.2 and Torchvision 0.17.2. The paper, response, and release
metadata must record the actually resolved environment and name
`resnet50.ram_in1k`; changing the import path, torchvision version, or timm
weight recipe creates a different experiment.

Both routes use Adam with network learning rate `5e-5` and weight decay zero.
HyperDG uses a separate Adam optimizer for local/global curvature parameters,
with learning rate `1e-4` in learned-curvature conditions and zero in fixed
conditions, and clips network and curvature gradient norms to 5. DomainBed ERM
uses its native single Adam optimizer and does not apply this clipping. These
method-specific optimizer differences are part of the executed treatments.

Batch normalization is active (`freeze_bn=False`), and the two algorithm routes
update its running statistics differently. HyperDG sends the three source
minibatches through the featurizer separately. After the optimizer step, its
training-accuracy diagnostic calls `self.predict(x)` once more for each source
while the model remains in training mode; `torch.no_grad()` suppresses
gradients but does not suppress BatchNorm running-statistic updates. Native
DomainBed ERM concatenates the source minibatches and performs one featurizer
pass. Consequently, FullPerDomain minus StandardERM is a whole-package utility
comparison, not an isolated causal estimate of curvature conditioning.
NoVariance minus SharedLearnable retains the same HyperDG route on both sides,
so this BatchNorm/update-path difference does not confound the primary
domain-specificity contrast.

## 5. Implementation caveats that constrain interpretation

### 5.1 The local curvature-gradient term is disconnected and inactive

For each source loader, the implementation computes the forward scale through
one indexing path:

```python
curv_val = curvatures[env_idx]
tangent_local = feats / sqrt(curv_val)
```

It separately creates `curv_param = self.log_curvatures[env_idx]` and asks
`autograd.grad(sup_loss, curv_param, allow_unused=True)`. That newly indexed view
is not the tensor view used to construct `curv_val`, so autograd returns `None`.
The code converts `None` to zero. As a result, `loss_local_flat` is logged as
exactly zero and `delta_local_flat=0.1` has no effect in these runs. This is an
implementation defect, not evidence that curvature sensitivity is naturally
zero.

### 5.2 The norm proxy algebraically cancels curvature

The so-called sharpness term forms

```text
norms = ||h(x) / sqrt(c)||,
proxy = Var(norms) * c.
```

Away from the inactive numerical clamp,

```text
Var(||h|| / sqrt(c)) * c = Var(||h||).
```

Thus the term regularizes raw feature-norm dispersion but has no substantive
curvature dependence. It is neither a curvature-sharpness measurement nor a
SAM inner maximization. It may still change the encoder through its feature
gradient, so it cannot simply be deleted from only one condition.

### 5.3 HyperDG trains with scaling but predicts without it

The HyperDG supervised and global losses pass scaled features to the classifier
during `update()`. Its `predict()` method instead applies the classifier to
unscaled encoder features. All logged source and target checkpoint accuracies,
and the `train_acc` diagnostic, call this unscaled prediction route. The learned
source or global curvature is therefore not applied at target inference.

This train/evaluation mismatch is part of every HyperDG-path legacy result and
every planned faithful SharedLearnable result. With a linear head, feature scale
can also be partly absorbed by classifier weights, weakening identification of
the curvature scalar. The results support claims about this training-time
scaling-and-regularization package, not a learned target hyperbolic geometry.

### 5.4 Curvature indices are source-loader slots

With one target held out, the trainer passes three minibatches to HyperDG in the
remaining environment order. `curvature_domain_0`, `_1`, and `_2` refer to those
three source-loader positions. Their named dataset domains change when the
held-out target changes. Any curvature trajectory figure must reconstruct that
mapping; the raw index alone is not a stable named-domain identifier across
leave-one-domain-out runs.

### 5.5 Target metrics were logged, but are excluded from the new selector

The historical trainer evaluated and stored target `in` and `out` accuracy at
every checkpoint. The reviewer-round selector demonstrably excludes the target
key, averages only the three source `out` keys, and breaks ties by earliest
step. The defensible statement is:

> Target-domain accuracy is not used by the frozen checkpoint-selection rule.

The stronger statement that target data were “never used for selection” across
all historical exploration cannot be established from these logs and must not
be made without a separate decision-history audit.

## 6. What must not be silently changed

HyperDG2 configs still execute the parent checkout's `scripts/train.py`,
`src/sftdg/algorithms/hyperdg.py`, and vendored DomainBed files. Editing those
parent files changes the treatment even if no HyperDG2 config changes. At this
snapshot, the relevant anchors are:

```text
git HEAD: ca8aabaf2a8aeebfcf54b19dcac0989c41fcc362
9cf636c42c710050e8b6cbce59bfc345ae9f6c06df308f3b8cb8dc0a0ec03fc9  src/sftdg/algorithms/hyperdg.py
ca3e7fac357ec2113998dffc1e9d9f435a12fb471c8d1a065c5b8f08b9b766b9  scripts/train.py
046f722577b48b596ae31714148f864a6688eb0e90810784960835e389c19bac  third_party/domainbed/domainbed/networks.py
16ef2adbd5fd8f3f4ca8cc5893d1d6a390fe09760655c3c12fb4cb4b2046475b  third_party/domainbed/domainbed/datasets.py
9e97b524733056bb8e6a8c604701a54b4f1841adb366258d3623cb402e1809ee  HyperDG2/manifests/inventory.csv
```

The git commit alone is insufficient if any relevant file is dirty; hashes and
the final release commit must accompany the raw metrics archive.

The following changes invalidate direct reuse or require a new, separately
named and balanced grid:

- repairing the disconnected local term only for SharedLearnable;
- replacing the curvature-canceling norm proxy or calling it SAM;
- applying a learned/shared/global curvature during evaluation;
- changing the backbone, pretrained weight recipe, augmentation, batch
  normalization mode, dropout, classifier, optimizer, clipping, learning rate,
  weight decay, batch size, update count, or checkpoint cadence;
- making source validation deterministic only for the new conditions;
- changing the 80/20 split, seed hash, environment ordering, target index, or
  data copy;
- selecting checkpoints with target accuracy, switching the primary endpoint
  from target `out` to target `in`, or making final-step accuracy primary after
  seeing results;
- relabeling ZeroPenalty as StandardERM or using it to fill StandardERM cells;
- retaining the incomplete TerraIncognita shared-curvature run;
- retuning only one condition or dataset using held-out-target performance; or
- reusing the older seed-0 SharedLearnable results in the confirmatory matrix
  despite their software-environment mismatch.

A scientifically useful corrected implementation is possible, but it must be
versioned as a new treatment and rerun across all relevant comparators. Corrected
and legacy HyperDG-path cells must not be pooled in one condition.

## 7. Confirmatory and exploratory evidence boundaries

The frozen reviewer-round protocol has two confirmatory contrasts:

1. **Utility:** FullPerDomain minus StandardERM.
2. **Domain specificity:** NoVariance minus SharedLearnable.

For each contrast, paired run-seed differences are first averaged within each
held-out target. A run seed changes the 80/20 split, initialization, minibatch
order, and augmentation, so it is not a pure optimization repeat. The four
target-domain effects, not the 16 target-by-run-seed cells, are the per-dataset
inferential units. At suite level, the four benchmark means are the inference
clusters because leave-one-domain-out targets within a benchmark reuse source
data. The source-selected target `out` endpoint and the revision-stage 1.0
percentage-point smallest effect of practical interest are primary.

At this audit snapshot, neither confirmatory contrast is complete:

- StandardERM has 0/64 cells at the frozen snapshot, so utility is not yet
  estimable.
- Clean SharedLearnable has 0/64 cells at the frozen snapshot, so domain
  specificity is not yet estimable.

No partial mean, seed-0 shared-curvature result, or currently populated row in
`condition_summary.csv` may support a confirmatory superiority, non-inferiority,
equivalence, or “no effect” claim. The complete balanced grid and declared
domain-clustered analysis are required first.

The following are secondary or exploratory:

- FullPerDomain minus ZeroPenalty, Frozen, or NoVariance;
- FullPerDomain minus SharedLearnable as a parameterization diagnostic;
- SharedLearnable minus StandardERM and ZeroPenalty minus StandardERM as
  protocol checks;
- legacy component-removal, variance-weight, and any non-inventoried
  shared-curvature diagnostic runs;
- target learning curves, which are post-hoc diagnostics and never select a
  checkpoint;
- update-4999 and target-`in` outcomes;
- 0.5- and 2.0-point equivalence-margin sensitivities;
- curvature trajectories, norm/proxy decompositions, timing, Poincare
  visualizations, MERU, DomainNet, and WILDS/Camelyon artifacts outside this
  exact grid.

Failure to reject a null hypothesis is not equivalence. The primary
equivalence question is the source-selected suite NoVariance-minus-Shared
macro; equivalence at the revision-stage margin requires its entire 90%
dataset-clustered interval to lie within `[-1,+1]` percentage point. This is a
claim about the equal-dataset/equal-target macro, not every target or the
methods in general. With four target clusters per dataset and four dataset
clusters at suite level, either exact two-sided sign-flip test has only 16
assignments and a minimum attainable p-value of 0.125; uncertainty and limited
power must be reported rather than hidden.

Finally, the source-selection code supports the claim that target metrics do
not enter the selector. It does not retrospectively prove that authors never
viewed target curves during earlier development. The response should use that
precise evidence boundary.

## 8. Release gate

Before any reviewer-facing result is treated as complete:

1. rerun `HyperDG2/scripts/validate_review_round2.py` and require zero errors;
2. require exactly 64 complete cells for each of all six conditions;
3. retain every new evidence path and status in `inventory.csv`, while keeping
   excluded legacy paths in the audit narrative rather than using them as
   completed cells;
4. run `HyperDG2/scripts/analyze_review_round2.py` without
   `--allow-incomplete`;
5. archive raw configs, logs, checkpoints or checkpoint hashes, code commit,
   file hashes, package versions, and failure history;
6. report every named target with all four run-seed values, mean, and SD, while
   separating between-target variation from within-target combined
   split-and-optimization run-seed variation; and
7. keep the confirmatory, sensitivity, and exploratory labels fixed when
   writing the paper and response.
