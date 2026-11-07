# A Controlled Evaluation of Curvature-Inspired Scalar Feature Scaling for Domain Generalization

Sunny Gupta and Amit Sethi, Indian Institute of Technology Bombay

Transactions on Machine Learning Research · Submission #10466

- [OpenReview discussion and submission record](https://openreview.net/forum?id=p3Yq1sU7qo)
- [Camera-ready manuscript](paper/manuscript.pdf)
- [Supplementary material](paper/supplementary.pdf)
- [Original HyperDG study and code](https://github.com/sunnyinAI/HyperDG)

The study evaluates learned scalar rescaling of Euclidean features during training for domain generalization. It compares six training conditions on PACS, VLCS, OfficeHome, and TerraIncognita, with four held-out targets and four seeds per dataset.

## Reproduce the camera-ready results

The repository contains the training code, all 384 run configurations and original metric logs, and the analysis scripts, tables, and figures. The [experiment protocol](HyperDG2/EXPERIMENT_PROTOCOL.md) describes the six conditions and the source-validation checkpoint selection rule. The [pre-completion implementation audit](HyperDG2/IMPLEMENTATION_AND_EVIDENCE_AUDIT.md) documents the executed objective and its limitations.

### Recompute the tables from the original logs

Run from the repository root with Python 3.10 or newer. This uses the logged evaluations and requires no datasets, GPU, or model weights.

```sh
python -m pip install numpy pandas scipy "matplotlib>=3.9" pyyaml
python HyperDG2/scripts/analyze_review_round2.py --metrics-only --output-dir reproduced
python HyperDG2/scripts/render_review_results.py --input-dir reproduced --output-dir reproduced/rendered
python HyperDG2/scripts/plot_review_round2.py --input-dir reproduced --output-dir reproduced/figures
python HyperDG2/scripts/plot_curvature_trajectories_all_seeds.py --input-dir reproduced \
  --figure-dir reproduced/figures --data-dir reproduced/trajectory-data
```

Compare the recomputed results with the [archived tables](HyperDG2/outputs/reviewer_round2/rendered/RESULTS_READY.md). The `--metrics-only` path validates the logged evaluation schedule and scores. The default analysis also checks for the saved final checkpoint of every run.

### Rerun training

Use a Linux machine with an NVIDIA GPU and create the environment from [`reproducibility/environment.yml`](reproducibility/environment.yml). Obtain the datasets under their respective terms using the [DomainBed data instructions](https://github.com/facebookresearch/DomainBed/tree/b93c22a1cfc3b2428398272c1a116c8de1f4139e#download-the-datasets). Place the datasets at `data/PACS/`, `data/VLCS/`, `data/terra_incognita/`, and `data/officehome_clean/office_home/`, preserving the domain and class subdirectories. The effective backbone is the pretrained timm model `resnet50.ram_in1k`. Training constructs a pretrained torchvision ResNet-50 first, then replaces it with the timm model. Both downloads require network access if absent from the caches. The [recorded runtime](reproducibility/recorded-runtime.json) lists the original package versions, dataset file counts, and both weight hashes. The [dataset inventory](reproducibility/dataset-files.tsv.gz) records the relative filenames and byte sizes used in the study so a local copy can be checked against the original inventory.

Use a separate checkout for training because it writes new logs to the same paths as the archived logs.

```sh
conda env create -f reproducibility/environment.yml
conda activate hyperdg-plus
export PYTHONPATH=src:third_party/domainbed
python HyperDG2/scripts/run_manifest.py --manifest HyperDG2/manifests/full_grid.txt --gpu 0
python HyperDG2/scripts/analyze_review_round2.py --output-dir reproduced
```

The bundled DomainBed code is from commit `b93c22a1cfc3b2428398272c1a116c8de1f4139e` and retains its [MIT license](third_party/domainbed/LICENSE). The manifest can be split into disjoint queues for multiple GPUs. The archived SHA-256 manifest records the original configs, logs, and checkpoints. Fresh training is expected to produce different file hashes. Dataset images and original model checkpoints are not distributed here.

### Build the paper and supplement

The [LaTeX sources](paper/source) include their figure and table inputs. With TeX Live and `latexmk` installed, run:

```sh
cd paper/source
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex supplementary.tex
```
