#!/usr/bin/env python3
"""Batch photo culling for nested folder structures.

Processes a directory tree, culling each folder with images independently
while preserving the folder structure in the output.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from image_loader import ImageLoader, SUPPORTED_EXTENSIONS
from culler import CullingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_image_folders(source_dir: Path) -> List[Tuple[Path, int]]:
    """Find all folders containing images.

    Args:
        source_dir: Root directory to search.

    Returns:
        List of (folder_path, image_count) tuples, sorted by path.
    """
    image_folders = []
    extensions = SUPPORTED_EXTENSIONS | {ext.upper() for ext in SUPPORTED_EXTENSIONS}

    # Check root folder
    root_images = sum(
        1 for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if root_images > 0:
        image_folders.append((source_dir, root_images))

    # Walk subdirectories
    for folder in sorted(source_dir.rglob("*")):
        if not folder.is_dir():
            continue
        # Skip hidden and cache folders
        if any(part.startswith('.') for part in folder.parts):
            continue

        image_count = sum(
            1 for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if image_count > 0:
            image_folders.append((folder, image_count))

    return image_folders


def compute_target(
    image_count: int,
    ratio: float,
    min_count: int,
    max_count: int
) -> int:
    """Compute target image count for a folder.

    Args:
        image_count: Number of images in folder.
        ratio: Target ratio (e.g., 0.1 for 10%).
        min_count: Minimum images to keep.
        max_count: Maximum images to keep.

    Returns:
        Target count, respecting min/max bounds.
    """
    target = int(image_count * ratio)
    target = max(target, min_count)
    target = min(target, max_count, image_count)
    return target


def process_folder(
    source_folder: Path,
    output_folder: Path,
    target_count: int,
    similarity_threshold: float,
    batch_size: int,
    use_llm: bool
) -> bool:
    """Process a single folder through the culling pipeline.

    Args:
        source_folder: Folder containing images.
        output_folder: Output destination.
        target_count: Number of images to select.
        similarity_threshold: Duplicate detection threshold.
        batch_size: CLIP batch size.
        use_llm: Whether to use Ollama.

    Returns:
        True if successful, False on error.
    """
    try:
        pipeline = CullingPipeline(
            source_dir=source_folder,
            output_dir=output_folder,
            target_count=target_count,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size,
            cache_dir=source_folder / ".cache",
            use_llm=use_llm,
            analyze_only=False,
            resume=True  # Always try to resume
        )
        pipeline.run()
        return True
    except Exception as e:
        logger.error(f"Failed to process {source_folder}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch photo culling for nested folder structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with 10% ratio (default)
  python batch_culler.py --source ./photos --output ./selected

  # Keep exactly 10 per folder
  python batch_culler.py --source ./photos --output ./selected --min-per-folder 10 --max-per-folder 10

  # Keep 20% with minimum 5 per folder
  python batch_culler.py --source ./photos --output ./selected --ratio 0.2 --min-per-folder 5

  # Flatten output to single folder
  python batch_culler.py --source ./photos --output ./selected --flat
"""
    )

    parser.add_argument(
        "--source", "-s",
        type=Path,
        required=True,
        help="Source directory containing photos (with optional subfolders)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for selected photos"
    )
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.1,
        help="Target ratio of images to keep (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--min-per-folder",
        type=int,
        default=10,
        help="Minimum images to keep per folder (default: 10)"
    )
    parser.add_argument(
        "--max-per-folder",
        type=int,
        default=1000,
        help="Maximum images to keep per folder (default: 1000)"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Flatten output to single folder instead of preserving structure"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Ollama for composition analysis"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold for duplicates (default: 0.92)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for CLIP processing (default: 16)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate source
    if not args.source.exists():
        logger.error(f"Source directory not found: {args.source}")
        sys.exit(1)

    # Find all folders with images
    logger.info(f"Scanning {args.source} for image folders...")
    image_folders = find_image_folders(args.source)

    if not image_folders:
        logger.error("No folders with images found")
        sys.exit(1)

    # Calculate targets and show plan
    total_source = sum(count for _, count in image_folders)
    total_target = 0

    print(f"\n{'Folder':<50} {'Images':>8} {'Target':>8}")
    print("-" * 70)

    processing_plan = []
    for folder, count in image_folders:
        # Compute relative path for output
        if folder == args.source:
            rel_path = Path(".")
        else:
            rel_path = folder.relative_to(args.source)

        if args.flat:
            output_folder = args.output
        else:
            output_folder = args.output / rel_path

        target = compute_target(
            count,
            args.ratio,
            args.min_per_folder,
            args.max_per_folder
        )
        total_target += target

        # Display path relative to source
        display_path = str(rel_path) if rel_path != Path(".") else "(root)"
        print(f"{display_path:<50} {count:>8} {target:>8}")

        processing_plan.append((folder, output_folder, count, target))

    print("-" * 70)
    print(f"{'TOTAL':<50} {total_source:>8} {total_target:>8}")
    print(f"\nWill process {len(processing_plan)} folder(s)")

    if args.dry_run:
        print("\n[DRY RUN] No files will be processed.")
        sys.exit(0)

    # Process each folder
    print()
    successful = 0
    failed = 0

    for folder, output_folder, count, target in tqdm(processing_plan, desc="Processing folders"):
        rel_path = folder.relative_to(args.source) if folder != args.source else Path(".")
        logger.info(f"Processing {rel_path} ({count} images -> {target})")

        if process_folder(
            folder,
            output_folder,
            target,
            args.similarity_threshold,
            args.batch_size,
            args.use_llm
        ):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Batch processing complete!")
    print(f"  Successful: {successful}/{len(processing_plan)} folders")
    if failed > 0:
        print(f"  Failed: {failed} folders")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
