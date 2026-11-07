# Frozen protocol for the Paper 10466 reviewer round

**Frozen:** 23 July 2026, before inspecting any new standard-ERM results or
seeds 1--3 of the new SharedLearnable grid.

## 1. Scope

The study is a controlled evaluation of the executed scalar
curvature-conditioned feature-scaling objective.  It is not a test of
hyperbolic representation learning in general, a Lorentz classifier, geodesic
prediction, or Sharpness-Aware Minimization.

The four benchmark suites are PACS, VLCS, OfficeHome, and TerraIncognita.  Each
contains four named environments.  Every environment is held out once, and the
remaining three are sources.  Seeds are 0, 1, 2, and 3.

The primary outcome remains accuracy on the held-out target environment's 20%
`out` split so that the new conditions match the submitted 256-run protocol.
Target `in` accuracy is retained in the compact run table as a sensitivity
outcome.

## 2. Conditions

The completed six-condition matrix is:

1. **StandardERM:** unmodified DomainBed `ERM`, supervised cross-entropy only,
   with no curvature variables, scaling, or auxiliary terms.
2. **ZeroPenalty:** HyperDG code path with all auxiliary coefficients and
   curvature learning set to zero.
3. **FullPerDomain:** one learned source curvature per source-domain slot plus
   the full submitted auxiliary objective.
4. **Frozen:** per-domain curvatures fixed at initialization while the remaining
   auxiliary objective stays active.
5. **NoVariance:** learned per-domain curvatures with the explicit curvature
   dispersion coefficient set to zero.
6. **SharedLearnable:** one learned local/source curvature shared by all source
   domains, plus the separate learned global scalar that is also present in the
   per-domain conditions.  Its local curvature-dispersion loss is structurally
   zero; all other HyperDG terms are retained.

The two confirmatory contrasts are:

- **Utility:** FullPerDomain minus StandardERM.
- **Domain specificity:** NoVariance minus SharedLearnable.

NoVariance is the clean comparator for SharedLearnable because neither
condition has an operative between-curvature dispersion loss.  Full minus
ZeroPenalty, Full minus NoVariance, Full minus Frozen, Full minus Shared, and
ZeroPenalty minus StandardERM are secondary mechanistic or protocol checks.

## 3. Matched training protocol

All conditions use:

- the same environment partitions and `seed_hash(seed, environment)` splits;
- the same seeds and paired held-out targets;
- `data_augmentation=True`;
- `resnet18=False`, `resnet50_augmix=True`, and `resnet_dropout=0`;
- the checkout's actual `timm` `resnet50.ram_in1k` feature extractor initialized
  from ImageNet weights;
- a linear classifier;
- Adam, backbone/classifier learning rate `5e-5`, weight decay 0;
- batch size 32 per source domain;
- 5,000 updates;
- evaluation every 200 updates and at update 4999;
- the same source-validation selection rule.

HyperDG-specific curvature variables use Adam at `1e-4`, and its network and
curvature gradients are clipped at norm 5 as in the submitted implementation.
Canonical DomainBed ERM has no curvature optimizer and uses its unmodified ERM
update.  This method-specific difference is disclosed; the optimization
budget, backbone, augmentation, learning rate, weight decay, and selection
protocol remain matched.

Batch normalization is not frozen.  The HyperDG update feeds each source
domain through the feature extractor separately, then performs a second
no-gradient feature-extractor pass to calculate its training-accuracy
diagnostic; both passes update BatchNorm running statistics in training mode.
Native DomainBed ERM instead concatenates the source minibatches for one
feature-extractor pass.  Thus FullPerDomain minus StandardERM estimates the
utility of the complete released training package, including this update-path
difference, and is not a curvature-only causal contrast.  The primary
NoVariance minus SharedLearnable comparison is unaffected by this caveat
because both conditions use the same HyperDG update route.

The current DomainBed dataset object assigns augmentation to source
environments before their 80/20 split.  Consequently, source `out` validation
examples use stochastic source augmentation.  This behavior is inherited by
all conditions and is not changed for only the new arms.  It is reported as a
protocol limitation.

## 4. Target-excluding model selection

For method \(m\), dataset \(j\), held-out target \(d\), seed \(s\), source
environment \(r\), and checkpoint \(t\), let \(V_{m j d s r t}\) be source
`out` validation accuracy.  The checkpoint is

\[
t^*_{m j d s}
=
\arg\max_t
\frac{1}{3}\sum_{r\ne d}V_{m j d s r t}.
\]

Source environments receive equal weight.  The held-out target key is excluded
from the score.  An exact tie selects the earliest checkpoint.  Only after
\(t^*\) is fixed is held-out target accuracy read from the corresponding log
record.

Source-validation-selected target accuracy is primary.  Update-4999 target
accuracy is a frozen revision-stage sensitivity analysis.  Source `out`
validation is evaluated once per logged checkpoint with the inherited
stochastic source transform; it is not a repeated Monte Carlo estimate.
Target curves are generated post hoc for diagnosis and never enter selection.

The existing trainer did evaluate and log the target at every checkpoint.
Therefore the precise defensible claim is that **target accuracy is not used by
the checkpoint-selection rule**.  The paper should not make a stronger
historical claim about all exploratory decisions unless the authors can audit
and substantiate it.

## 5. Estimands

Let \(Y_{m j d s}\) be source-selected target accuracy in percentage points.
For treatment \(A\) and comparator \(B\),

\[
g_{jds}^{A,B}=Y_{A j d s}-Y_{B j d s},
\qquad
\bar g_{jd}^{A,B}=\frac{1}{4}\sum_s g_{jds}^{A,B}.
\]

The dataset estimand is the equal-target average

\[
\Delta_j^{A,B}=\frac{1}{4}\sum_d \bar g_{jd}^{A,B}.
\]

It describes the four named targets in that benchmark; it is not an estimate
over an unrestricted population of possible domains.

The cross-benchmark macro estimand is

\[
\Delta_{\mathrm{macro}}^{A,B}
=
\frac{1}{4}\sum_j\Delta_j^{A,B},
\]

which weights each dataset and then each target within a dataset equally.

## 6. Uncertainty and multiplicity

A run seed changes the 80/20 split through
`seed_hash(seed, environment_index)`, as well as initialization, minibatch
order, and augmentation randomness.  The four run seeds are paired across
methods within a target; they are repeated run-level measurements, not four
additional target-domain draws and not pure optimization replicates.  For each
dataset, inference operates on the four values \(\bar g_{jd}\):

\[
\widehat\Delta_j
\pm
t_{3,.975}
\frac{\operatorname{SD}_d(\bar g_{jd})}{\sqrt{4}}.
\]

The analysis also enumerates all \(2^4=16\) target-cluster sign flips.  All
seed differences from one target flip together.  The minimum two-sided p-value
is therefore 0.125.  An exhaustive four-cluster bootstrap is reported only as
a sensitivity analysis because four-cluster percentile intervals are unstable.

For the suite macro, the four values \(\Delta_j\), one per benchmark dataset,
are the primary inference clusters.  This accounts for the fact that the four
leave-one-domain-out targets within a benchmark are nested and reuse source
data.  The suite report uses a \(t_3\) interval, all \(2^4=16\) dataset-level
sign flips, and an exhaustive four-dataset bootstrap sensitivity.  A
stratified calculation that treats all 16 target effects as independent within
dataset strata is retained only in `suite_domain_sensitivity.csv`; it is not
used for a confirmatory suite claim.

The two source-selected suite-macro confirmatory contrasts receive Holm
correction as one family of two.  The eight per-dataset estimates (two
contrasts by four datasets) are named-target analyses reported with a separate
Holm-8 sensitivity adjustment; they are not a second confirmatory family.
Secondary mechanistic contrasts are clearly labeled.

## 7. Practical-effect decision rules

The frozen revision-stage smallest effect of practical interest is
\(\delta=1.0\) accuracy percentage point.  It was selected after the submitted
results were known but before the added grid completed, and is not described as
preregistered or universally canonical.  Sensitivity margins are 0.5 and 2.0
points.

Equivalence uses two one-sided tests at alpha 0.05, equivalently requiring the
relevant 90% clustered interval to lie wholly inside
\([-\delta,+\delta]\).  The primary equivalence question is the
source-selected suite macro for NoVariance minus SharedLearnable at the 1.0 pp
margin.  Per-dataset TOSTs, the Full-minus-StandardERM TOST, alternate margins,
and final-step TOSTs are sensitivity analyses.  Failure to satisfy the
condition is **inconclusive**, not evidence of equivalence.

For the domain-specificity question, a complementary non-inferiority statement
uses SharedLearnable minus NoVariance.  SharedLearnable is non-inferior within
one point only if its one-sided 95% lower bound exceeds -1.0 point.

Allowed claims are:

- positive utility only if Full minus StandardERM has a lower confidence bound
  above zero under the declared analysis;
- practical macro equivalence only if the complete 90% interval is inside the
  SESOI; this conclusion concerns the equal-dataset/equal-target average, not
  every target or the methods in general;
- a ruled-out benefit of at least one point only if the corresponding one-sided
  upper bound is below +1.0;
- otherwise, an explicit inconclusive result with the compatible effect range.

Statistical evidence that a mean differs from zero and practical equivalence
within a nonzero SESOI are separate questions and can both be true.

## 8. Variability reporting

Every named target is reported with all four run-seed values plus mean +/- SD.
The SD combines split, initialization, minibatch-order, and augmentation
variation.  Dataset summaries separately report:

\[
SD_{\mathrm{target}}=SD_d(\bar Y_{d\cdot})
\]

and the root-mean-square within-target run-seed SD.  A balanced two-way
descriptive decomposition additionally reports run-seed-mean variation and
target-by-run-seed residual variation.  The same decomposition is applied to
paired method effects in `effect_variance_decomposition.csv`; it does not
identify pure split versus pure optimization components.

No pooled SD over the 16 run-seed-by-target cells is used as if it represented
one source of uncertainty.  Target `in` condition summaries and primary
contrast sensitivities are written to `target_in_sensitivity.csv`.

## 9. Failure and inclusion policy

A run is complete only when:

- `metrics.jsonl` contains exactly the 25 expected evaluation records at
  updates 200, 400, ..., 4800, and 4999;
- every record contains all four environments' `in` and `out` accuracy keys;
  and
- `model_last.pt` exists.

Incomplete runs are excluded from confirmatory analysis and rerun from the
beginning.  Inclusion does not depend on target accuracy.  Failures and missing
cells remain visible in `missing_runs.json`; the final analysis exits nonzero
until the balanced matrix is complete.

No legacy seed-0 SharedLearnable cell is reused for confirmatory inference.
Twelve otherwise complete cells were generated under an older
Torch/Torchvision environment, three were generated near the environment
transition, and one TerraIncognita cell is incomplete.  All 64 SharedLearnable
cells are rerun under the same resolved environment as the new StandardERM
grid.  Legacy paths remain documented only as excluded exploratory provenance.

## 10. Frozen implementation boundary

The HyperDG computation is not silently repaired for only the new condition.
Doing so would change the treatment and require rerunning every HyperDG-path
condition.  The implementation audit is reported separately, including the
inactive local curvature-gradient term, the curvature-canceling norm proxy,
the train/evaluation scaling mismatch, and the actual backbone resolution.

This protocol is designed to answer the present review.  It cannot guarantee
acceptance, and it cannot turn four target environments into a well-powered
random-domain population study.
