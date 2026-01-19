"""Aesthetic scoring using LAION's predictor with CLIP embeddings."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import torch
import torch.nn as nn

from embeddings import get_device

logger = logging.getLogger(__name__)

# URL for LAION aesthetic predictor v2 linear weights
AESTHETIC_MODEL_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac+logos+ava1-l14-linearMSE.pth"
)


class AestheticMLP(nn.Module):
    """Simple MLP for aesthetic prediction from CLIP embeddings."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticScorer:
    """Score images using LAION aesthetic predictor v2.

    This uses the linear model trained on AVA, LAION-Logos, and SAC datasets.
    It takes CLIP ViT-L/14 embeddings (768-dim) as input.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize aesthetic scorer.

        Args:
            model_path: Path to model weights. Downloads if None.
            device: Torch device. Auto-selects if None.
        """
        self.device = device or get_device()
        self.model_path = model_path
        self._model = None

        logger.info(f"Aesthetic scorer will use device: {self.device}")

    def _download_model(self) -> Path:
        """Download model weights if needed."""
        import urllib.request
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "aesthetic_predictor"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file = cache_dir / "sac_logos_ava1_l14_linear.pth"

        if not model_file.exists():
            logger.info("Downloading LAION aesthetic predictor weights...")
            urllib.request.urlretrieve(AESTHETIC_MODEL_URL, model_file)
            logger.info(f"Downloaded to {model_file}")

        return model_file

    @property
    def model(self) -> AestheticMLP:
        """Lazy load aesthetic model."""
        if self._model is None:
            model_path = self.model_path or self._download_model()

            self._model = AestheticMLP(input_dim=768)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model = self._model.to(self.device)
            self._model.eval()

            logger.info("Loaded LAION aesthetic predictor")

        return self._model

    def score_from_embeddings(
        self,
        embeddings: np.ndarray,
        batch_size: int = 256
    ) -> np.ndarray:
        """Score images from their CLIP embeddings.

        Args:
            embeddings: Array of shape (N, 768) CLIP embeddings.
            batch_size: Processing batch size.

        Returns:
            Array of aesthetic scores (raw model output, typically 1-10 range).
        """
        scores = []

        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(self.device)

            with torch.no_grad():
                batch_scores = self.model(batch_tensor)
                scores.append(batch_scores.cpu().numpy())

        return np.concatenate(scores).flatten()

    def score_batch_from_embeddings(
        self,
        paths: List[Path],
        embeddings: Dict[Path, np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[Path, float]:
        """Score images using precomputed CLIP embeddings.

        This is the efficient path - reuses embeddings from deduplication.

        Args:
            paths: List of image paths to score.
            embeddings: Dictionary mapping paths to CLIP embeddings.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Dictionary mapping paths to aesthetic scores (1-10 scale).
        """
        # Filter to paths with embeddings
        valid_paths = [p for p in paths if p in embeddings]

        if not valid_paths:
            logger.warning("No valid embeddings to score")
            return {}

        # Stack embeddings in order
        embedding_matrix = np.stack([embeddings[p] for p in valid_paths])

        # Score all at once
        logger.info(f"Scoring {len(valid_paths)} images...")
        raw_scores = self.score_from_embeddings(embedding_matrix)

        # Normalize to 1-10 range
        # Raw LAION scores are approximately 1-10 but can exceed bounds
        normalized_scores = self._normalize_scores(raw_scores)

        # Build result dict
        results = {}
        for i, path in enumerate(valid_paths):
            results[path] = float(normalized_scores[i])
            if progress_callback:
                progress_callback(i + 1, len(valid_paths))

        return results

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 1-10 range.

        The LAION predictor outputs roughly 1-10 scores, but they can
        sometimes exceed these bounds. We clip and scale to ensure
        consistent 1-10 output.
        """
        # Clip to reasonable range first
        clipped = np.clip(scores, 1.0, 10.0)

        # Scores are already roughly 1-10, so just ensure bounds
        return clipped

    def get_statistics(self, scores: Dict[Path, float]) -> Dict[str, float]:
        """Compute statistics for a set of scores.

        Args:
            scores: Dictionary mapping paths to scores.

        Returns:
            Dictionary with mean, std, min, max, median.
        """
        if not scores:
            return {}

        values = list(scores.values())
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
