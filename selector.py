"""Final image selection and ranking based on combined scores."""

import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from deduplication import DuplicateCluster

logger = logging.getLogger(__name__)


@dataclass
class ScoredImage:
    """An image with all its scores."""
    path: Path
    aesthetic_score: float
    composition_score: Optional[float]
    final_score: float
    cluster_id: Optional[int]
    selected: bool = False
    rank: Optional[int] = None


class ImageSelector:
    """Select and rank images based on combined scores."""

    def __init__(
        self,
        aesthetic_weight: float = 0.6,
        composition_weight: float = 0.4
    ):
        """Initialize selector.

        Args:
            aesthetic_weight: Weight for aesthetic scores (when using LLM).
            composition_weight: Weight for composition scores (when using LLM).
        """
        self.aesthetic_weight = aesthetic_weight
        self.composition_weight = composition_weight

    def compute_final_scores(
        self,
        paths: List[Path],
        aesthetic_scores: Dict[Path, float],
        composition_scores: Optional[Dict[Path, float]] = None,
        clusters: Optional[List[DuplicateCluster]] = None
    ) -> List[ScoredImage]:
        """Compute final scores for all images.

        Args:
            paths: List of image paths.
            aesthetic_scores: Dictionary of aesthetic scores.
            composition_scores: Optional dictionary of composition scores.
            clusters: Optional list of duplicate clusters.

        Returns:
            List of ScoredImage objects with computed final scores.
        """
        # Build cluster lookup
        cluster_lookup: Dict[Path, int] = {}
        if clusters:
            for cluster in clusters:
                for member in cluster.members:
                    cluster_lookup[member] = cluster.cluster_id

        results = []
        for path in paths:
            aesthetic = aesthetic_scores.get(path, 5.0)
            composition = composition_scores.get(path) if composition_scores else None

            # Compute final score
            if composition is not None:
                final = (
                    self.aesthetic_weight * aesthetic +
                    self.composition_weight * composition
                )
            else:
                final = aesthetic

            results.append(ScoredImage(
                path=path,
                aesthetic_score=aesthetic,
                composition_score=composition,
                final_score=final,
                cluster_id=cluster_lookup.get(path)
            ))

        return results

    def select_top_images(
        self,
        scored_images: List[ScoredImage],
        target_count: int
    ) -> List[ScoredImage]:
        """Select top N images by final score.

        Args:
            scored_images: List of scored images.
            target_count: Number of images to select.

        Returns:
            Sorted list of selected images with ranks assigned.
        """
        # Sort by final score descending
        sorted_images = sorted(
            scored_images,
            key=lambda x: x.final_score,
            reverse=True
        )

        # Select top N
        selected = sorted_images[:target_count]

        # Assign ranks and mark as selected
        for i, img in enumerate(selected):
            img.rank = i + 1
            img.selected = True

        return selected

    def copy_with_ranking(
        self,
        selected_images: List[ScoredImage],
        output_dir: Path,
        dry_run: bool = False
    ) -> List[Path]:
        """Copy selected images to output directory with rank prefixes.

        Args:
            selected_images: List of selected ScoredImage objects with ranks.
            output_dir: Directory to copy images to.
            dry_run: If True, don't actually copy files.

        Returns:
            List of output paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []
        rank_width = len(str(len(selected_images)))

        for img in selected_images:
            if img.rank is None:
                continue

            # Build output filename with rank prefix
            rank_str = str(img.rank).zfill(rank_width)
            original_name = img.path.name
            output_name = f"{rank_str}_{original_name}"
            output_path = output_dir / output_name

            if not dry_run:
                shutil.copy2(img.path, output_path)
                logger.debug(f"Copied {img.path.name} -> {output_name}")

            output_paths.append(output_path)

        logger.info(f"{'Would copy' if dry_run else 'Copied'} "
                   f"{len(output_paths)} images to {output_dir}")

        return output_paths

    def generate_report(
        self,
        all_images: List[ScoredImage],
        output_path: Path
    ) -> None:
        """Generate CSV report with all scores.

        Args:
            all_images: List of all scored images.
            output_path: Path for output CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort by final score for the report
        sorted_images = sorted(
            all_images,
            key=lambda x: x.final_score,
            reverse=True
        )

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'rank',
                'filename',
                'path',
                'aesthetic_score',
                'composition_score',
                'final_score',
                'cluster_id',
                'selected'
            ])

            # Data rows
            for i, img in enumerate(sorted_images):
                writer.writerow([
                    img.rank if img.selected else '',
                    img.path.name,
                    str(img.path),
                    f"{img.aesthetic_score:.2f}",
                    f"{img.composition_score:.2f}" if img.composition_score else '',
                    f"{img.final_score:.2f}",
                    img.cluster_id if img.cluster_id is not None else '',
                    'yes' if img.selected else 'no'
                ])

        logger.info(f"Generated report: {output_path}")

    def select_and_export(
        self,
        paths: List[Path],
        aesthetic_scores: Dict[Path, float],
        output_dir: Path,
        target_count: int,
        composition_scores: Optional[Dict[Path, float]] = None,
        clusters: Optional[List[DuplicateCluster]] = None,
        analyze_only: bool = False
    ) -> List[ScoredImage]:
        """Complete selection pipeline: score, select, copy, report.

        Args:
            paths: List of image paths to consider.
            aesthetic_scores: Dictionary of aesthetic scores.
            output_dir: Output directory for selected images and report.
            target_count: Number of images to select.
            composition_scores: Optional composition scores.
            clusters: Optional duplicate clusters.
            analyze_only: If True, generate report but don't copy images.

        Returns:
            List of all scored images.
        """
        # Compute scores
        all_scored = self.compute_final_scores(
            paths, aesthetic_scores, composition_scores, clusters
        )

        # Select top images
        selected = self.select_top_images(all_scored, target_count)

        # Copy selected images (unless analyze_only)
        if not analyze_only:
            self.copy_with_ranking(selected, output_dir)

        # Generate report
        report_path = output_dir / "report.csv"
        self.generate_report(all_scored, report_path)

        return all_scored
