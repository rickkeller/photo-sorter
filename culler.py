#!/usr/bin/env python3
"""Photo culling pipeline CLI.

Reduces photo libraries by removing duplicates and selecting
the best images based on aesthetic and composition scores.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from image_loader import ImageLoader
from embeddings import CLIPEmbeddingGenerator
from deduplication import ImageDeduplicator, DuplicateCluster
from aesthetic import AestheticScorer
from composition import CompositionAnalyzer
from selector import ImageSelector
from checkpoint import CheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CullingPipeline:
    """Orchestrates the photo culling pipeline."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        target_count: int = 400,
        similarity_threshold: float = 0.92,
        batch_size: int = 16,
        cache_dir: Optional[Path] = None,
        use_llm: bool = False,
        llm_candidates: int = 800,
        analyze_only: bool = False,
        resume: bool = False
    ):
        """Initialize the culling pipeline.

        Args:
            source_dir: Directory containing source images.
            output_dir: Directory for output images and report.
            target_count: Number of images to select.
            similarity_threshold: Cosine similarity threshold for duplicates.
            batch_size: Batch size for CLIP processing.
            cache_dir: Directory for embedding cache.
            use_llm: Whether to use Ollama for composition analysis.
            llm_candidates: Number of top candidates for LLM analysis.
            analyze_only: Generate report only, don't copy images.
            resume: Resume from checkpoint if available.
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_count = target_count
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.use_llm = use_llm
        self.llm_candidates = llm_candidates
        self.analyze_only = analyze_only
        self.resume = resume

        # Set up cache directory
        self.cache_dir = cache_dir or (self.source_dir / ".cache")

        # Initialize components
        self.image_loader = ImageLoader()
        self.embedding_generator = CLIPEmbeddingGenerator(
            cache_dir=self.cache_dir / "embeddings",
            batch_size=batch_size
        )
        self.deduplicator = ImageDeduplicator(similarity_threshold)
        self.aesthetic_scorer = AestheticScorer()
        self.composition_analyzer = CompositionAnalyzer() if use_llm else None
        self.selector = ImageSelector()

        # Checkpoint management
        checkpoint_path = self.cache_dir / "checkpoint.json"
        self.checkpoint = CheckpointManager(checkpoint_path)

        # Pipeline state
        self.images: List[Path] = []
        self.embeddings: Dict[Path, np.ndarray] = {}
        self.unique_images: List[Path] = []
        self.clusters: List[DuplicateCluster] = []
        self.aesthetic_scores: Dict[Path, float] = {}
        self.composition_scores: Dict[Path, float] = {}

    def run(self) -> None:
        """Execute the complete culling pipeline."""
        logger.info(f"Starting culling pipeline")
        logger.info(f"  Source: {self.source_dir}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Target: {self.target_count} images")

        # Handle resume
        if self.resume and self.checkpoint.exists():
            state = self.checkpoint.load()
            if state:
                logger.info("Resuming from checkpoint:")
                logger.info(self.checkpoint.get_summary())
        else:
            self.checkpoint.create(self.source_dir, self.output_dir)

        # Execute pipeline phases
        self._discover_images()
        self._generate_embeddings()
        self._run_deduplication()
        self._score_aesthetic()

        if self.use_llm:
            self._score_composition()

        self._select_images()

        # Clean up checkpoint on success
        self.checkpoint.delete()
        logger.info("Pipeline complete!")

    def _discover_images(self) -> None:
        """Phase 1: Discover all images in source directory."""
        if self.checkpoint.should_skip_discovery():
            self.images = [Path(p) for p in self.checkpoint.state.discovered_images]
            logger.info(f"Skipping discovery (checkpoint): {len(self.images)} images")
            return

        logger.info("Discovering images...")
        self.images = self.image_loader.discover_images(self.source_dir)

        if not self.images:
            logger.error("No images found in source directory")
            sys.exit(1)

        logger.info(f"Found {len(self.images)} images")
        self.checkpoint.set_discovered(self.images)

    def _generate_embeddings(self) -> None:
        """Phase 2: Generate CLIP embeddings for all images."""
        if self.checkpoint.should_skip_embeddings():
            # Load embeddings from cache
            logger.info("Loading cached embeddings...")
            self.embeddings, _ = self.embedding_generator.get_embeddings_for_paths(
                self.images, self.image_loader
            )
            logger.info(f"Loaded {len(self.embeddings)} embeddings from cache")
            return

        # Check for partial progress
        to_process = self.images
        if self.resume:
            completed = set(self.checkpoint.state.embeddings_complete)
            to_process = [p for p in self.images if str(p) not in completed]
            if len(to_process) < len(self.images):
                logger.info(f"Resuming embeddings: {len(self.images) - len(to_process)} "
                          f"already complete")

        logger.info(f"Generating embeddings for {len(to_process)} images...")

        with tqdm(total=len(to_process), desc="Generating embeddings") as pbar:
            def progress(current, total):
                pbar.update(current - pbar.n)

            self.embeddings, failed = self.embedding_generator.get_embeddings_for_paths(
                to_process, self.image_loader, progress
            )

        # Update checkpoint
        for path in self.embeddings:
            self.checkpoint.add_embedding(path)
        for path in failed:
            self.checkpoint.add_embedding_failure(path)

        # Also load any previously completed embeddings
        if self.resume:
            prev_embeddings, _ = self.embedding_generator.get_embeddings_for_paths(
                [Path(p) for p in self.checkpoint.state.embeddings_complete],
                self.image_loader
            )
            self.embeddings.update(prev_embeddings)

        self.checkpoint.set_embeddings_complete()
        logger.info(f"Generated {len(self.embeddings)} embeddings "
                   f"({len(failed)} failed)")

    def _run_deduplication(self) -> None:
        """Phase 3: Find and cluster duplicate images."""
        if self.checkpoint.should_skip_dedup():
            self.unique_images = self.checkpoint.get_unique_images()
            # Reconstruct clusters from checkpoint
            cluster_data = self.checkpoint.state.duplicate_clusters
            self.clusters = [
                DuplicateCluster(
                    cluster_id=c['cluster_id'],
                    representative=Path(c['representative']),
                    members=[Path(m) for m in c['members']]
                )
                for c in cluster_data
            ]
            logger.info(f"Skipping dedup (checkpoint): {len(self.unique_images)} unique")
            return

        logger.info("Running deduplication...")

        # First pass without aesthetic scores (just to find clusters)
        self.unique_images, self.clusters = self.deduplicator.find_duplicates(
            self.images, self.embeddings
        )

        # Save to checkpoint
        cluster_data = [
            {
                'cluster_id': c.cluster_id,
                'representative': str(c.representative),
                'members': [str(m) for m in c.members]
            }
            for c in self.clusters
        ]
        self.checkpoint.set_dedup_results(self.unique_images, cluster_data)

        logger.info(f"Found {len(self.unique_images)} unique images "
                   f"({len(self.clusters)} duplicate clusters)")

    def _score_aesthetic(self) -> None:
        """Phase 4: Score images with LAION aesthetic predictor."""
        if self.checkpoint.should_skip_aesthetic():
            self.aesthetic_scores = self.checkpoint.get_aesthetic_scores()
            logger.info(f"Skipping aesthetic (checkpoint): "
                       f"{len(self.aesthetic_scores)} scores")
            return

        logger.info("Scoring aesthetics...")

        with tqdm(total=len(self.unique_images), desc="Aesthetic scoring") as pbar:
            def progress(current, total):
                pbar.update(current - pbar.n)

            self.aesthetic_scores = self.aesthetic_scorer.score_batch_from_embeddings(
                self.unique_images, self.embeddings, progress
            )

        self.checkpoint.set_aesthetic_scores(self.aesthetic_scores)

        stats = self.aesthetic_scorer.get_statistics(self.aesthetic_scores)
        logger.info(f"Aesthetic scores: mean={stats['mean']:.2f}, "
                   f"std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")

        # Re-run deduplication with aesthetic scores to pick best representatives
        if self.clusters:
            logger.info("Updating cluster representatives based on aesthetic scores...")
            self.unique_images, self.clusters = self.deduplicator.find_duplicates(
                self.images, self.embeddings, self.aesthetic_scores
            )

    def _score_composition(self) -> None:
        """Phase 5: Score top candidates with Ollama composition analysis."""
        if not self.use_llm or not self.composition_analyzer:
            return

        if self.checkpoint.should_skip_composition():
            self.composition_scores = self.checkpoint.get_composition_scores()
            logger.info(f"Skipping composition (checkpoint): "
                       f"{len(self.composition_scores)} scores")
            return

        if not self.composition_analyzer.check_availability():
            logger.warning("Ollama not available, skipping composition analysis")
            return

        logger.info(f"Analyzing composition for top {self.llm_candidates} candidates...")

        with tqdm(total=min(self.llm_candidates, len(self.unique_images)),
                  desc="Composition analysis") as pbar:
            def progress(current, total):
                pbar.update(current - pbar.n)

            self.composition_scores = self.composition_analyzer.analyze_top_candidates(
                self.unique_images,
                self.aesthetic_scores,
                self.llm_candidates,
                self.image_loader,
                progress
            )

        self.checkpoint.set_composition_scores(self.composition_scores)
        logger.info(f"Analyzed composition for {len(self.composition_scores)} images")

    def _select_images(self) -> None:
        """Phase 6: Select top images and generate output."""
        logger.info(f"Selecting top {self.target_count} images...")

        all_scored = self.selector.select_and_export(
            paths=self.unique_images,
            aesthetic_scores=self.aesthetic_scores,
            output_dir=self.output_dir,
            target_count=self.target_count,
            composition_scores=self.composition_scores if self.use_llm else None,
            clusters=self.clusters,
            analyze_only=self.analyze_only
        )

        selected = [img for img in all_scored if img.selected]

        self.checkpoint.set_selection_complete()

        if self.analyze_only:
            logger.info(f"Analysis complete. Report saved to {self.output_dir}/report.csv")
        else:
            logger.info(f"Selected {len(selected)} images")
            logger.info(f"Output saved to {self.output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Photo culling pipeline - reduce photos by removing duplicates "
                    "and selecting the best images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python culler.py --source /path/to/photos --output /path/to/selected --target 400

  # With LLM composition analysis
  python culler.py --source /path/to/photos --output /path/to/selected --target 400 --use-llm

  # Analysis only (no copy)
  python culler.py --source /path/to/photos --output /path/to/report --analyze-only

  # Resume interrupted run
  python culler.py --source /path/to/photos --output /path/to/selected --resume
"""
    )

    parser.add_argument(
        "--source", "-s",
        type=Path,
        required=True,
        help="Source directory containing photos"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for selected photos and report"
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=400,
        help="Target number of images to select (default: 400)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Ollama for composition analysis"
    )
    parser.add_argument(
        "--llm-candidates",
        type=int,
        default=800,
        help="Number of top candidates for LLM analysis (default: 800)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Generate report only, don't copy images"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for CLIP processing (default: 16)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold for duplicates (default: 0.92)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory for embedding cache (default: source/.cache)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.source.exists():
        logger.error(f"Source directory not found: {args.source}")
        sys.exit(1)

    if not args.source.is_dir():
        logger.error(f"Source is not a directory: {args.source}")
        sys.exit(1)

    if args.target < 1:
        logger.error("Target must be at least 1")
        sys.exit(1)

    # Create and run pipeline
    pipeline = CullingPipeline(
        source_dir=args.source,
        output_dir=args.output,
        target_count=args.target,
        similarity_threshold=args.similarity_threshold,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        use_llm=args.use_llm,
        llm_candidates=args.llm_candidates,
        analyze_only=args.analyze_only,
        resume=args.resume
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("\nInterrupted. Progress saved to checkpoint.")
        logger.info("Resume with --resume flag.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.info("Resume with --resume flag.")
        raise


if __name__ == "__main__":
    main()
