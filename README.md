# Photo Sorter

Photo Sorter culls large photo libraries. It removes near-duplicates and keeps the best-looking shots, ranked by an aesthetic quality model. Everything runs locally on your machine. No photos are uploaded anywhere.

Give it a folder of thousands of photos and a target (for example 400). It copies the top 400 into an output folder, ranked `01_`, `02_`, ..., and writes a `report.csv` with every score.

## Features

- **Duplicate removal.** Uses OpenAI's CLIP (ViT-L/14) to find visually similar images and keeps only the best one from each cluster.
- **Aesthetic scoring.** Ranks images with LAION's aesthetic predictor (a small model on top of the CLIP embeddings, trained on the AVA, SAC and LAION-Logos datasets).
- **Optional LLM composition check.** With `--use-llm`, a local [Ollama](https://ollama.com/) vision model (llava:13b) also scores composition.
- **HEIC support.** Handles Apple HEIC/HEIF alongside JPG and PNG.
- **CPU by default, GPU if you have one.** The default install is the CPU-only PyTorch build. If you install a CUDA or Apple MPS build, `--device auto` uses it.
- **Resume and caching.** Interrupted runs can resume, and embeddings are cached so re-runs are fast.

## Requirements

- Python 3.10 or newer
- Disk space:
  - About 2 GB for the Python packages with the default CPU install (the CPU-only torch wheel saves roughly 4 GB versus the CUDA build).
  - About 1.7 GB for the models, downloaded automatically on the first run (CLIP ViT-L/14 into the Hugging Face cache, plus the small LAION weights into `~/.cache/aesthetic_predictor/`).
  - If you choose the full CUDA (NVIDIA GPU) PyTorch instead, the virtual environment grows to about 5.9 GB.
- An internet connection for the first run (model downloads). Later runs work offline.
- Optional: [Ollama](https://ollama.com/) with a vision model, only if you use `--use-llm`.

## Install (default: CPU only)

```bash
git clone https://github.com/rickkeller/photo-sorter.git
cd photo-sorter

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` points pip at the CPU-only PyTorch index (`--index-url https://download.pytorch.org/whl/cpu`, with PyPI as the extra index for everything else). This is the smallest install and works on any OS. It's the right choice if you run Photo Sorter now and then.

> **About the large download.** PyTorch is the biggest dependency. The CPU-only build keeps the install small. The full CUDA build is about 5.9 GB installed. Either way, the first run also downloads about 1.7 GB of model weights. Make sure you have the disk space before you start.

## Optional: GPU acceleration

Only do this if you have the hardware and are OK with the bigger download. Install the default requirements first, then replace torch:

**NVIDIA GPU (CUDA 12.8), about 5.9 GB installed:**

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

**Apple Silicon (MPS).** Use the normal PyPI wheel, not the CPU index:

```bash
pip install --upgrade torch
```

Then run as usual. `--device auto` (the default) picks MPS first, then CUDA, then CPU. To force one, pass `--device cpu`, `--device cuda` or `--device mps`. Forcing `cuda` or `mps` with a torch build that doesn't support it stops with a clear error.

`batch_culler.py` has no `--device` flag. It always uses auto.

## Quickstart

Try it on a small folder first so you can see the output and how long it takes:

```bash
# Score everything and write report.csv only. Nothing is copied.
python culler.py --source ~/Pictures/some-trip --output ./report --analyze-only

# Copy the best 50 into ./selected, ranked 01_, 02_, ...
python culler.py --source ~/Pictures/some-trip --output ./selected --target 50
```

The first run spends a few minutes downloading models. On CPU, expect it to be noticeably slower than on a GPU for large libraries; the embedding cache makes later runs on the same folder much faster.

## Usage

### Select the top N

```bash
python culler.py --source /path/to/photos --output /path/to/selected --target 400
```

### With LLM composition analysis

Requires Ollama running locally (default `http://localhost:11434`) with a vision model pulled:

```bash
ollama pull llava:13b
python culler.py --source /path/to/photos --output /path/to/selected --target 400 --use-llm
```

The final score becomes 0.6 × aesthetic + 0.4 × LLM composition. Only the top `--llm-candidates` (default 800) are sent to the LLM. If Ollama isn't reachable, the run continues without it.

### Analysis only

```bash
python culler.py --source /path/to/photos --output /path/to/report --analyze-only
```

### Resume an interrupted run

Press Ctrl+C at any time; progress is saved. Then:

```bash
python culler.py --source /path/to/photos --output /path/to/selected --resume
```

### CLI options (`culler.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--source`, `-s` | required | Folder containing photos |
| `--output`, `-o` | required | Folder for the selected photos and `report.csv` |
| `--target`, `-t` | 400 | Number of images to select |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `mps`. Auto uses MPS, then CUDA, then CPU |
| `--use-llm` | off | Add Ollama composition scoring |
| `--llm-candidates` | 800 | How many top images the LLM scores |
| `--analyze-only` | off | Write the report only, don't copy images |
| `--resume` | off | Resume from the checkpoint if there is one |
| `--batch-size` | 16 | CLIP batch size (lower it if you run out of memory) |
| `--similarity-threshold` | 0.92 | Cosine similarity at or above which images count as duplicates |
| `--cache-dir` | `{source}/.cache` | Where embeddings are cached |
| `--verbose`, `-v` | off | Verbose logging |

## Batch processing

For a library organised in subfolders (one per trip or event), `batch_culler.py` culls each folder on its own and mirrors the folder structure in the output.

```bash
# Keep 10% of each folder (minimum 10, maximum 1000)
python batch_culler.py --source /path/to/photos --output /path/to/selected

# Exactly 10 per folder
python batch_culler.py --source /path/to/photos --output /path/to/selected --min-per-folder 10 --max-per-folder 10

# Preview the plan without processing
python batch_culler.py --source /path/to/photos --output /path/to/selected --dry-run
```

| Option | Default | Description |
|--------|---------|-------------|
| `--source`, `-s` | required | Folder with subfolders of photos |
| `--output`, `-o` | required | Output folder (structure preserved) |
| `--ratio`, `-r` | 0.1 | Share of images to keep per folder |
| `--min-per-folder` | 10 | Minimum kept per folder |
| `--max-per-folder` | 1000 | Maximum kept per folder |
| `--flat` | off | Put everything in one output folder |
| `--similarity-threshold` | 0.92 | Duplicate threshold |
| `--batch-size` | 16 | CLIP batch size |
| `--dry-run` | off | Show the plan only |
| `--use-llm` | off | Add Ollama composition scoring |
| `--verbose`, `-v` | off | Verbose logging |

```
source/                          output/
├── Trip-Italy/                  ├── Trip-Italy/
│   ├── Rome/ (500 photos)       │   ├── Rome/ (10 best)
│   └── Venice/ (300 photos)     │   └── Venice/ (10 best)
└── Trip-Japan/ (1000 photos)    └── Trip-Japan/ (10 best)
```

## Output

```
output/
├── 01_IMG_1234.jpg    # highest ranked
├── 02_DSC_5678.jpg
├── ...
└── report.csv         # every image and its scores
```

`report.csv` columns: `rank` (empty if not selected), `filename`, `path`, `aesthetic_score` (1–10), `composition_score` (1–10, only with `--use-llm`), `final_score`, `cluster_id` (duplicate cluster, if any), `selected`.

Your original photos are never moved or modified; selected ones are copied.

## Configuration

There are no environment variables or config files. Everything is set with CLI flags. Files Photo Sorter writes outside the output folder:

- `{source}/.cache/` (or `--cache-dir`): embedding cache and checkpoint. Safe to delete.
- `~/.cache/huggingface/`: the CLIP model (standard Hugging Face cache; set `HF_HOME` to move it).
- `~/.cache/aesthetic_predictor/`: the LAION aesthetic weights.

## How it works

1. **Discovery.** Finds `.jpg`, `.jpeg`, `.png`, `.heic` and `.heif` files in the source folder.
2. **Embedding.** Computes CLIP ViT-L/14 embeddings (cached to disk).
3. **Deduplication.** Groups images with cosine similarity ≥ the threshold (Union-Find) and keeps the best of each group.
4. **Aesthetic scoring.** Scores the unique images with the LAION predictor, reusing the CLIP embeddings.
5. **Composition (optional).** Sends the top candidates to the Ollama vision model.
6. **Selection.** Ranks by final score and copies the top N with rank prefixes.

Known limits: the aesthetic model tends to score scenic landscapes higher than people or documentary shots, so a mixed library may get more landscapes than you expect. Very wide panoramas score lower because CLIP looks at a centre square crop. Darker exposure variants of the same shot aren't always caught as duplicates.

## Troubleshooting

- **The install is huge / downloading gigabytes.** You probably installed the CUDA build of torch. The default `pip install -r requirements.txt` uses the CPU-only wheel. To switch back: `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu`.
- **First run seems stuck.** It's downloading about 1.7 GB of models. Run with `-v` to see progress.
- **`CUDA was requested but this torch build does not have it`** (or the MPS equivalent). You passed `--device cuda`/`mps` but have the CPU wheel installed. Follow the GPU section above, or use `--device auto`.
- **On a Mac, it runs on CPU.** The CPU index wheel doesn't include MPS. Run `pip install --upgrade torch` (PyPI wheel).
- **Out of memory.** Lower `--batch-size` (for example 4 or 8).
- **HEIC files are skipped or fail.** Make sure `pillow-heif` installed correctly (`pip install pillow-heif`).
- **`--use-llm` does nothing.** Check that Ollama is running (`ollama list`) and that `llava:13b` (or another llava tag) is pulled. Warnings are printed if the model isn't found.
- **Too many near-duplicates kept.** Lower `--similarity-threshold` (for example 0.88). Too many distinct shots merged: raise it.

## Future ideas

A lighter install path using ONNX Runtime instead of PyTorch is a possible future option. It would shrink the download further, but it isn't implemented yet.

## License

MIT. See [LICENSE](LICENSE).
