# Photo Sorter

A photo culling pipeline that reduces large photo libraries by removing duplicates and selecting the best images based on aesthetic quality scores.

## Features

- **CLIP-based deduplication** - Uses OpenAI's CLIP model to find visually similar images via cosine similarity clustering
- **Aesthetic scoring** - Ranks images using LAION's aesthetic predictor trained on AVA dataset
- **Optional LLM composition analysis** - Uses Ollama vision models to evaluate composition (rule of thirds, leading lines, etc.)
- **HEIC support** - Handles Apple's HEIC format alongside JPG/PNG
- **MPS acceleration** - Optimized for Apple Silicon with automatic fallback to CUDA/CPU
- **Resume capability** - Checkpoint system allows resuming interrupted runs
- **Embedding cache** - Disk caching prevents re-processing on subsequent runs

## Installation

```bash
# Clone the repository
git clone https://github.com/rickkeller/photo-sorter.git
cd photo-sorter

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Select the top 400 images from a photo library:

```bash
python culler.py --source /path/to/photos --output /path/to/selected --target 400
```

### With LLM Composition Analysis

Add Ollama-based composition scoring (requires [Ollama](https://ollama.ai/) with a vision model):

```bash
python culler.py --source /path/to/photos --output /path/to/selected --target 400 --use-llm
```

### Analysis Only

Generate a report without copying images:

```bash
python culler.py --source /path/to/photos --output /path/to/report --analyze-only
```

### Resume Interrupted Run

```bash
python culler.py --source /path/to/photos --output /path/to/selected --resume
```

## Batch Processing

For directories with multiple subfolders (e.g., photos organized by trip or event), use `batch_culler.py` to process each folder independently while preserving the directory structure.

### Basic Batch Usage

Process all subfolders, keeping 10% of images per folder (minimum 10):

```bash
python batch_culler.py --source /path/to/photos --output /path/to/selected
```

### Fixed Count Per Folder

Keep exactly 10 images per folder:

```bash
python batch_culler.py --source /path/to/photos --output /path/to/selected --min-per-folder 10 --max-per-folder 10
```

### Preview Mode

See what would be processed without actually running:

```bash
python batch_culler.py --source /path/to/photos --output /path/to/selected --dry-run
```

### Batch CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source`, `-s` | required | Source directory with subfolders |
| `--output`, `-o` | required | Output directory (structure preserved) |
| `--ratio`, `-r` | 0.1 | Target ratio of images to keep (10%) |
| `--min-per-folder` | 10 | Minimum images per folder |
| `--max-per-folder` | 1000 | Maximum images per folder |
| `--flat` | false | Flatten output to single folder |
| `--dry-run` | false | Preview without processing |
| `--use-llm` | false | Enable Ollama composition analysis |

### Batch Output

Output mirrors the source folder structure:

```
source/                          output/
├── Trip-Italy/                  ├── Trip-Italy/
│   ├── Rome/ (500 photos)       │   ├── Rome/ (10 best)
│   └── Venice/ (300 photos)     │   └── Venice/ (10 best)
└── Trip-Japan/ (1000 photos)    └── Trip-Japan/ (10 best)
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source`, `-s` | required | Source directory containing photos |
| `--output`, `-o` | required | Output directory for selected photos and report |
| `--target`, `-t` | 400 | Number of images to select |
| `--use-llm` | false | Enable Ollama composition analysis |
| `--llm-candidates` | 800 | Number of top candidates for LLM analysis |
| `--analyze-only` | false | Generate report only, don't copy images |
| `--resume` | false | Resume from checkpoint if available |
| `--batch-size` | 16 | Batch size for CLIP processing |
| `--similarity-threshold` | 0.92 | Cosine similarity threshold for duplicates |
| `--cache-dir` | `{source}/.cache` | Directory for embedding cache |
| `--verbose`, `-v` | false | Enable verbose logging |

## Output

Selected images are copied to the output directory with rank prefixes:

```
output/
├── 01_IMG_1234.jpg    # Highest ranked
├── 02_DSC_5678.jpg
├── 03_photo.png
├── ...
└── report.csv         # Full scoring report
```

The `report.csv` contains all processed images with columns:
- `rank` - Selection rank (empty if not selected)
- `filename` - Original filename
- `path` - Full path to source image
- `aesthetic_score` - LAION aesthetic score (1-10)
- `composition_score` - LLM composition score (1-10, if enabled)
- `final_score` - Combined score used for ranking
- `cluster_id` - Duplicate cluster ID (if part of a cluster)
- `selected` - Whether image was selected

## How It Works

1. **Discovery** - Finds all JPG, PNG, and HEIC images in source directory
2. **Embedding** - Generates CLIP ViT-L/14 embeddings for each image (cached to disk)
3. **Deduplication** - Clusters similar images using Union-Find on cosine similarity matrix
4. **Aesthetic Scoring** - Scores unique images using LAION predictor (reuses CLIP embeddings)
5. **Composition Analysis** - (Optional) Analyzes top candidates with Ollama vision model
6. **Selection** - Ranks by final score, copies top N images with rank prefixes

## Requirements

- Python 3.10+
- ~3GB disk space for CLIP model weights
- Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU

## License

MIT
