"""Checkpoint management for resume capability."""

import json
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """State of the culling pipeline for checkpoint/resume."""

    # Metadata
    source_dir: str
    output_dir: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Discovery phase
    discovered_images: List[str] = field(default_factory=list)
    discovery_complete: bool = False

    # Embedding phase
    embeddings_complete: List[str] = field(default_factory=list)  # Paths with embeddings
    embeddings_failed: List[str] = field(default_factory=list)
    embeddings_phase_complete: bool = False

    # Deduplication phase
    unique_images: List[str] = field(default_factory=list)
    duplicate_clusters: List[Dict[str, Any]] = field(default_factory=list)
    dedup_complete: bool = False

    # Scoring phase
    aesthetic_scores: Dict[str, float] = field(default_factory=dict)
    aesthetic_complete: bool = False
    composition_scores: Dict[str, float] = field(default_factory=dict)
    composition_complete: bool = False

    # Selection phase
    selection_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manage checkpoint files for pipeline resume capability."""

    def __init__(self, checkpoint_path: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint JSON file.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self._state: Optional[PipelineState] = None
        self._save_interval = 100  # Save after every N embeddings

    @property
    def state(self) -> Optional[PipelineState]:
        """Current pipeline state."""
        return self._state

    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_path.exists()

    def load(self) -> Optional[PipelineState]:
        """Load checkpoint from file.

        Returns:
            PipelineState if checkpoint exists and is valid, None otherwise.
        """
        if not self.exists():
            return None

        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
            self._state = PipelineState.from_dict(data)
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            return self._state
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Invalid checkpoint file: {e}")
            return None

    def create(self, source_dir: Path, output_dir: Path) -> PipelineState:
        """Create new checkpoint state.

        Args:
            source_dir: Source directory for images.
            output_dir: Output directory.

        Returns:
            New PipelineState object.
        """
        self._state = PipelineState(
            source_dir=str(source_dir),
            output_dir=str(output_dir)
        )
        self.save()
        return self._state

    def save(self) -> None:
        """Save current state to checkpoint file atomically."""
        if self._state is None:
            return

        self._state.updated_at = datetime.now().isoformat()

        # Atomic write via temp file + rename
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=self.checkpoint_path.parent,
            suffix='.tmp',
            delete=False
        ) as f:
            json.dump(self._state.to_dict(), f, indent=2)
            temp_path = Path(f.name)

        temp_path.rename(self.checkpoint_path)
        logger.debug(f"Checkpoint saved to {self.checkpoint_path}")

    def delete(self) -> None:
        """Delete checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {self.checkpoint_path}")
        self._state = None

    # Phase update methods

    def set_discovered(self, images: List[Path]) -> None:
        """Update discovery phase."""
        if self._state:
            self._state.discovered_images = [str(p) for p in images]
            self._state.discovery_complete = True
            self.save()

    def add_embedding(self, path: Path) -> None:
        """Record a completed embedding."""
        if self._state:
            path_str = str(path)
            if path_str not in self._state.embeddings_complete:
                self._state.embeddings_complete.append(path_str)

            # Save periodically
            if len(self._state.embeddings_complete) % self._save_interval == 0:
                self.save()

    def add_embedding_failure(self, path: Path) -> None:
        """Record a failed embedding."""
        if self._state:
            path_str = str(path)
            if path_str not in self._state.embeddings_failed:
                self._state.embeddings_failed.append(path_str)

    def set_embeddings_complete(self) -> None:
        """Mark embedding phase as complete."""
        if self._state:
            self._state.embeddings_phase_complete = True
            self.save()

    def set_dedup_results(
        self,
        unique_images: List[Path],
        clusters: List[Dict[str, Any]]
    ) -> None:
        """Update deduplication results."""
        if self._state:
            self._state.unique_images = [str(p) for p in unique_images]
            self._state.duplicate_clusters = clusters
            self._state.dedup_complete = True
            self.save()

    def set_aesthetic_scores(self, scores: Dict[Path, float]) -> None:
        """Update aesthetic scores."""
        if self._state:
            self._state.aesthetic_scores = {str(p): s for p, s in scores.items()}
            self._state.aesthetic_complete = True
            self.save()

    def set_composition_scores(self, scores: Dict[Path, float]) -> None:
        """Update composition scores."""
        if self._state:
            self._state.composition_scores = {str(p): s for p, s in scores.items()}
            self._state.composition_complete = True
            self.save()

    def set_selection_complete(self) -> None:
        """Mark selection as complete."""
        if self._state:
            self._state.selection_complete = True
            self.save()

    # Resume helper methods

    def get_images_needing_embeddings(self) -> List[Path]:
        """Get images that still need embeddings generated."""
        if not self._state:
            return []

        discovered = set(self._state.discovered_images)
        complete = set(self._state.embeddings_complete)
        failed = set(self._state.embeddings_failed)

        return [Path(p) for p in discovered - complete - failed]

    def get_aesthetic_scores(self) -> Dict[Path, float]:
        """Get saved aesthetic scores as Path keys."""
        if not self._state:
            return {}
        return {Path(p): s for p, s in self._state.aesthetic_scores.items()}

    def get_composition_scores(self) -> Dict[Path, float]:
        """Get saved composition scores as Path keys."""
        if not self._state:
            return {}
        return {Path(p): s for p, s in self._state.composition_scores.items()}

    def get_unique_images(self) -> List[Path]:
        """Get unique images from dedup phase."""
        if not self._state:
            return []
        return [Path(p) for p in self._state.unique_images]

    def should_skip_discovery(self) -> bool:
        """Check if discovery phase can be skipped."""
        return self._state is not None and self._state.discovery_complete

    def should_skip_embeddings(self) -> bool:
        """Check if embeddings phase can be skipped."""
        return self._state is not None and self._state.embeddings_phase_complete

    def should_skip_dedup(self) -> bool:
        """Check if deduplication can be skipped."""
        return self._state is not None and self._state.dedup_complete

    def should_skip_aesthetic(self) -> bool:
        """Check if aesthetic scoring can be skipped."""
        return self._state is not None and self._state.aesthetic_complete

    def should_skip_composition(self) -> bool:
        """Check if composition scoring can be skipped."""
        return self._state is not None and self._state.composition_complete

    def get_summary(self) -> str:
        """Get human-readable summary of checkpoint state."""
        if not self._state:
            return "No checkpoint"

        lines = [
            f"Checkpoint: {self.checkpoint_path}",
            f"  Source: {self._state.source_dir}",
            f"  Started: {self._state.started_at}",
            f"  Updated: {self._state.updated_at}",
            "",
            "Phases:",
            f"  Discovery: {'✓' if self._state.discovery_complete else '✗'} "
            f"({len(self._state.discovered_images)} images)",
            f"  Embeddings: {'✓' if self._state.embeddings_phase_complete else '✗'} "
            f"({len(self._state.embeddings_complete)}/{len(self._state.discovered_images)})",
            f"  Deduplication: {'✓' if self._state.dedup_complete else '✗'} "
            f"({len(self._state.unique_images)} unique)",
            f"  Aesthetic: {'✓' if self._state.aesthetic_complete else '✗'} "
            f"({len(self._state.aesthetic_scores)} scored)",
            f"  Composition: {'✓' if self._state.composition_complete else '✗'} "
            f"({len(self._state.composition_scores)} scored)",
            f"  Selection: {'✓' if self._state.selection_complete else '✗'}",
        ]
        return "\n".join(lines)
