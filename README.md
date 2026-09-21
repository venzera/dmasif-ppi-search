# dMaSIF PPI screen

A staged protein–protein interaction screening pipeline:

1. validate query and target structures against FASTA records;
2. generate dMaSIF-search surface embeddings;
3. optionally restrict the search to dMaSIF-site-predicted interface points;
4. retrieve complementary surface patches with FAISS;
5. geometrically verify shortlisted pairs with Open3D RANSAC and ICP;
6. export the top pairs and their sequences for AlphaFold2-multimer, AlphaFold3,
   Boltz, or another complex predictor.

This repository performs **candidate retrieval and geometric filtering**. It does
not claim that a shortlisted pair binds.

## Why installation is split

The original [FreyrS/dMaSIF](https://github.com/FreyrS/dMaSIF) code was tested
with Python 3.6/3.7, PyTorch 1.4/1.6, PyKeOps 1.4, PyTorch Geometric 1.5/1.6,
CUDA 10.x, and GCC 7/8. FAISS and current Open3D are easier to install in a
modern environment. Mixing both stacks in one environment is fragile.

Pre-trained weights are available in https://github.com/casperg92/MaSIF_colab

Use:

- `dmasif-legacy` for embedding and optional site scoring;
- `dmasif-search` for FAISS retrieval, RANSAC/ICP, and pair export.

The `.npz` files written between the two stages are the environment boundary.

## Requirements

- Linux is strongly recommended for GPU dMaSIF.
- An NVIDIA GPU and a system CUDA toolkit compatible with the selected PyTorch
  build are required for practical embedding.
- A C/C++ compiler is required because PyKeOps compiles kernels.
- CPU RAM must hold the filtered target embeddings. Large target libraries may
  require tens of gigabytes.
- A dMaSIF-search checkpoint compatible with the architecture below is required.
- A dMaSIF-site checkpoint is optional.

The checkpoints are **not distributed by this repository**. Confirm their
source and licence before redistribution.

## Installation

Clone this repository and upstream dMaSIF next to it:

```bash
git clone https://github.com/FreyrS/dMaSIF.git
git clone <THIS-REPOSITORY-URL> dmasif-ppi-screen
cd dmasif-ppi-screen
```

### 1. Legacy dMaSIF environment

The supplied environment reproduces the dependency era used by upstream
dMaSIF:

```bash
conda env create -f envs/dmasif-legacy.yml
conda activate dmasif-legacy
```

Install PyTorch Geometric extensions using wheels that match **PyTorch 1.6 and
your CUDA build**. For CUDA 10.2:

```bash
pip install \
  torch-scatter==2.0.5 \
  torch-sparse==0.6.8 \
  torch-cluster==1.5.8 \
  torch-spline-conv==1.2.0 \
  -f https://data.pyg.org/whl/torch-1.6.0+cu102.html
pip install torch-geometric==1.6.1
pip install -e . --no-deps
```

Verify PyTorch, CUDA and PyKeOps before starting a large run:

```bash
python - <<'PY'
import torch
import pykeops
print("torch", torch.__version__, "CUDA available:", torch.cuda.is_available())
pykeops.test_torch_bindings()
PY
```

Old CUDA binaries may not support recent GPU architectures. If this environment
cannot run on your GPU, start from the upstream dMaSIF Dockerfile or construct a
newer compatible PyTorch/PyKeOps stack and validate it on a small structure.
Do not silently change model architecture parameters.

### 2. Modern search environment

```bash
conda env create -f envs/search.yml
conda activate dmasif-search
pip install -e . --no-deps
python -c "import faiss, open3d; print('FAISS/Open3D OK')"
```

`faiss-cpu` is the default. Install a CUDA-compatible FAISS build separately
and pass `--gpu` to the search command if GPU FAISS is available.

## Input format

Four inputs are required:

```text
query_structures/       PDB or mmCIF query structures
target_structures/      PDB or mmCIF target structures
queries.fasta           query sequences
targets.fasta           target sequences
```

Every structure file stem must exactly match its FASTA identifier:

```text
query_structures/protein_A.pdb
>protein_A
MKK...
```

Identifiers must be unique. Directories are searched recursively by default.
Only the first whitespace-delimited token in each FASTA header is used.

## Pipeline

The examples below use:

```bash
export DMASIF_REPO=/path/to/dMaSIF
export SEARCH_CKPT=/path/to/dMaSIF_search_3layer_12A_16dim
export SITE_CKPT=/path/to/dMaSIF_site_3layer_16dims_12A_100sup_epoch71
export WORK=$PWD/work
```

### Step 1 — validate inputs and write manifests

This step can run in either environment:

```bash
python -m dmasif_screen.prepare \
  --query-structures query_structures \
  --target-structures target_structures \
  --query-fasta queries.fasta \
  --target-fasta targets.fasta \
  --work-dir "$WORK"
```

The command fails if structure and FASTA identifiers differ. Use
`--allow-missing` only when intentionally screening their intersection.

### Step 2 — generate dMaSIF-search embeddings

Activate `dmasif-legacy`:

```bash
conda activate dmasif-legacy

python -m dmasif_screen.embed \
  --manifest "$WORK/manifests/query.tsv" \
  --output-dir "$WORK/embeddings/query" \
  --dmasif-repo "$DMASIF_REPO" \
  --checkpoint "$SEARCH_CKPT"

python -m dmasif_screen.embed \
  --manifest "$WORK/manifests/target.tsv" \
  --output-dir "$WORK/embeddings/target" \
  --dmasif-repo "$DMASIF_REPO" \
  --checkpoint "$SEARCH_CKPT"
```

Defaults reproduce this pipeline's search model: three layers, 12 Å radius,
16-dimensional descriptors, 1 Å resolution, and 20-fold surface
supersampling. Surface clouds are capped at 2,000 points and structures with
fewer than 20 points are skipped.

Embedding is resumable. Existing `.npz` files are not recomputed. For a job
array, give every task a distinct zero-based `--shard-index` and a common
`--num-shards`.

### Step 3 — optional interface-site scoring

Site filtering reduces the search space but requires a compatible site
checkpoint:

```bash
python -m dmasif_screen.score_sites \
  --manifest "$WORK/manifests/query.tsv" \
  --embedding-dir "$WORK/embeddings/query" \
  --dmasif-repo "$DMASIF_REPO" \
  --checkpoint "$SITE_CKPT"

python -m dmasif_screen.score_sites \
  --manifest "$WORK/manifests/target.tsv" \
  --embedding-dir "$WORK/embeddings/target" \
  --dmasif-repo "$DMASIF_REPO" \
  --checkpoint "$SITE_CKPT"
```

The site and search networks produce different point clouds. Site
probabilities are transferred to search points by three-dimensional nearest
neighbour mapping. Inspect `site_nn_dist` in the `.npz` files when validating a
new checkpoint.

To skip this stage, pass `--site-threshold 0` in both Steps 4 and 5.

### Step 4 — FAISS descriptor retrieval

Activate the modern environment:

```bash
conda activate dmasif-search

python -m dmasif_screen.search \
  --query-embeddings "$WORK/embeddings/query" \
  --target-embeddings "$WORK/embeddings/target" \
  --output-dir "$WORK/search" \
  --site-threshold 0.5 \
  --z-threshold 3.0 \
  --topk-points 32 \
  --max-targets-per-query 500
```

The search uses the cross-descriptor dot products learned by dMaSIF:

```text
query.embedding_1 × target.embedding_2
query.embedding_2 × target.embedding_1
```

It does not use cosine or Euclidean distance. A null distribution is estimated
from 200,000 random query–target point pairs and matches are retained at
`z > 3`. `background.json` is saved and reused by geometric rescoring.

`max_z` and `sigmoid_logit` are shortlist scores, not validated binding
probabilities. `sigmoid_logit` is only the sigmoid of the maximum raw
descriptor dot product.

### Step 5 — RANSAC and ICP geometric filtering

```bash
python -m dmasif_screen.rescore \
  --search-dir "$WORK/search" \
  --query-embeddings "$WORK/embeddings/query" \
  --target-embeddings "$WORK/embeddings/target" \
  --output-dir "$WORK/rescored" \
  --top-n 50 \
  --workers 8
```

For each query, this command tests the 50 best retrieval candidates. Descriptor
matches are treated as candidate point correspondences. Open3D RANSAC searches
for one rigid transformation that makes several correspondences
simultaneously consistent, and point-to-plane ICP refines the result.

Recovered protocol values include a 1 Å RANSAC inlier radius, 2,000 iterations,
an edge-length ratio of 0.9, a 1.5 Å distance checker, a 90° normal checker and
1 Å point-to-plane ICP refinement. The modern Open3D API uses a confidence
criterion; this implementation uses 0.999 instead of the historical old-API
value of 500 validation steps.

The default normal-neighbourhood radius (3 Å), maximum neighbours (30), and
minimum correspondence count (3) were not recoverable from the deployed
configuration. They are explicit portable defaults and should be validated for
a new dataset. The final heuristic is:

```text
post-ICP correspondences / (1 + inlier_RMSD)
```

Surface normals are estimated from the saved point cloud; they are not the
analytical dMaSIF normals. PCA normal orientation is not guaranteed, so the
normal checker must not be described as proving that the two surfaces have
opposed normals. The output reports RANSAC inliers and post-ICP correspondences
separately and saves the final query-to-target 4×4 transformation. The score is
an uncalibrated ranking measure. All geometric parameters are exposed as
command-line options.

### Step 6 — select and export top pairs

The default ranking uses reciprocal-rank fusion of dMaSIF `max_z` and the
RANSAC score:

```bash
python -m dmasif_screen.select_pairs \
  --rescored-dir "$WORK/rescored" \
  --query-fasta queries.fasta \
  --target-fasta targets.fasta \
  --output-dir "$WORK/selected" \
  --top-n-per-query 3 \
  --rank-by rrf \
  --min-inliers 3
```

Useful filters include `--min-fitness`, `--max-rmsd`, and `--top-n-total`.
Outputs are:

```text
selected/pairs.tsv          scores, identifiers, lengths, and sequences
selected/pairs.fasta        all selected two-chain FASTA records
selected/pair_fastas/       one two-chain FASTA per pair
```

## Complex prediction handoff

The exported pair FASTAs can be supplied to:

- AlphaFold2-multimer/ColabFold;
- AlphaFold3;
- Boltz;
- OpenFold3;
- another complex-prediction system.

Use independent per-chain MSAs when taxonomy-based pairing is not biologically
meaningful, for example for host–pathogen pairs. Evaluate predicted complexes
with interface-aware confidence measures such as ipTM, PAE-derived metrics, or
ipSAE. The surface screen and a complex prediction remain computational
hypotheses and require biological validation.

## Reproducibility and limitations

- The embedding and site checkpoint architectures are loaded with
  `strict=True`; incompatible weights fail rather than being partially loaded.
- Input and output units are ångströms.
- Search is resumable per query; embedding is resumable per structure.
- FAISS `IndexFlatIP` is exact, but only the requested nearest point matches are
  considered.
- RANSAC is stochastic. Repeated runs can produce slightly different inlier
  counts.
- There is no atom-level clash filter in the current geometric stage.
- A high score does not establish binding, directionality, affinity, or
  physiological relevance.

## Citation

If you use this workflow, cite dMaSIF and the downstream tools used in your
analysis:

> Sverrisson F, Feydy J, Correia BE, Bronstein MM. Fast end-to-end learning on
> protein surfaces. CVPR (2021).

Also cite FAISS, Open3D, and the selected complex-prediction software.
