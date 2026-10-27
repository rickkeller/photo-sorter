# AGENTS.md

Notes for coding agents (and humans) working on this repo.

## What this is

Photo Sorter: a local Python CLI that dedups and ranks photos with CLIP ViT-L/14 embeddings plus the LAION aesthetic predictor, with optional Ollama (llava) composition scoring.

## Layout

All modules are flat at the repo root and run as scripts (no package, no pyproject yet).

| File | Role |
|------|------|
| `culler.py` | Main CLI and `CullingPipeline` (discovery → embed → dedup → score → select → copy) |
| `batch_culler.py` | Runs `CullingPipeline` once per subfolder; mirrors folder structure |
| `embeddings.py` | `CLIPEmbeddingGenerator`, disk cache, and device choice (`choose_device`, `get_device`) |
| `aesthetic.py` | LAION aesthetic MLP; downloads weights to `~/.cache/aesthetic_predictor/` |
| `deduplication.py` | Cosine-similarity clustering (Union-Find) |
| `composition.py` | Ollama client (`http://localhost:11434`, default model `llava:13b`) |
| `selector.py` | Final ranking and copy with rank prefixes, `report.csv` |
| `checkpoint.py` | JSON checkpoint for `--resume` |
| `image_loader.py` | Image discovery and loading, HEIC via pillow-heif |
| `requirements.txt` | Dependencies; uses the CPU-only torch index by default |

## Setup and run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # CPU-only torch wheel
python culler.py --help
python culler.py --source <dir> --output <dir> --analyze-only
```

The first run downloads about 1.7 GB of models. There is no test suite yet. For a smoke test, run `--analyze-only` on a handful of small images and check `report.csv`.

## Conventions

- Keep the CPU-only torch wheel as the default in `requirements.txt`. GPU support is opt-in: users replace torch themselves, and `--device auto` picks MPS, then CUDA, then CPU. Don't hard-pin a device, and don't make CUDA the default install.
- Never modify or move files in `--source`; only copy into `--output`.
- Keep everything local by default. Anything that sends photos off the machine must be opt-in.
- When adding a CLI flag, update the options tables in `README.md` (both scripts if relevant).
- Use `logging`, not `print`, except for the batch plan table.

## Don't commit

- Secrets, tokens or API keys of any kind.
- Photos, test images, outputs or caches (`test_images/`, `test_output/`, `.cache/`, `*.npy` are already in `.gitignore`).
- Model weights or virtual environments.
