"""CLIP embedding generation with disk caching and MPS support."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from image_loader import ImageLoader, get_image_hash

logger = logging.getLogger(__name__)

# CLIP model used for both deduplication and aesthetic scoring
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def get_device() -> torch.device:
    """Select best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CLIPEmbeddingGenerator:
    """Generate and cache CLIP embeddings for images."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        batch_size: int = 16,
        device: Optional[torch.device] = None
    ):
        """Initialize CLIP model and cache.

        Args:
            cache_dir: Directory for embedding cache. Creates if needed.
            batch_size: Number of images per batch.
            device: Torch device. Auto-selects if None.
        """
        self.device = device or get_device()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None

        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CLIP will use device: {self.device}")

    @property
    def model(self) -> CLIPModel:
        """Lazy load CLIP model."""
        if self._model is None:
            logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
            self._model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    @property
    def processor(self) -> CLIPProcessor:
        """Lazy load CLIP processor."""
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        return self._processor

    def _get_cache_path(self, image_path: Path) -> Optional[Path]:
        """Get cache file path for an image."""
        if not self.cache_dir:
            return None

        file_hash = get_image_hash(image_path)
        # Use first 2 chars as subdirectory for better filesystem performance
        subdir = self.cache_dir / file_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{file_hash}.npy"

    def _load_cached(self, image_path: Path) -> Optional[np.ndarray]:
        """Try to load embedding from cache."""
        cache_path = self._get_cache_path(image_path)
        if cache_path and cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        return None

    def _save_to_cache(self, image_path: Path, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        cache_path = self._get_cache_path(image_path)
        if cache_path:
            try:
                np.save(cache_path, embedding)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

    def generate_batch(
        self,
        images: List[Tuple[Path, Image.Image]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[Path, np.ndarray]:
        """Generate embeddings for a batch of images.

        Args:
            images: List of (path, PIL.Image) tuples.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Dictionary mapping paths to embedding arrays.
        """
        results = {}
        total = len(images)
        current_batch_size = self.batch_size

        i = 0
        while i < total:
            batch = images[i:i + current_batch_size]

            try:
                # Process batch
                batch_images = [img for _, img in batch]
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    # Normalize embeddings
                    embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
                    embeddings = embeddings.cpu().numpy()

                # Store results and cache
                for j, (path, _) in enumerate(batch):
                    results[path] = embeddings[j]
                    self._save_to_cache(path, embeddings[j])

                # Clear MPS cache to prevent OOM
                if self.device.type == "mps":
                    torch.mps.empty_cache()

                i += len(batch)
                if progress_callback:
                    progress_callback(min(i, total), total)

                # Reset batch size on success
                current_batch_size = self.batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_batch_size > 1:
                    # Reduce batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(f"OOM, reducing batch size to {current_batch_size}")
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    elif self.device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

        return results

    def get_embeddings_for_paths(
        self,
        paths: List[Path],
        image_loader: Optional[ImageLoader] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[Dict[Path, np.ndarray], List[Path]]:
        """Get embeddings for image paths, using cache when available.

        Args:
            paths: List of image paths.
            image_loader: ImageLoader instance. Creates one if None.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Tuple of (embeddings_dict, failed_paths).
        """
        if image_loader is None:
            image_loader = ImageLoader()

        embeddings = {}
        to_process = []
        failed = []

        # Check cache first
        for path in paths:
            cached = self._load_cached(path)
            if cached is not None:
                embeddings[path] = cached
            else:
                to_process.append(path)

        logger.info(f"Found {len(embeddings)} cached embeddings, "
                   f"need to generate {len(to_process)}")

        if not to_process:
            return embeddings, failed

        # Load images that need processing
        def load_progress(current, total):
            if progress_callback:
                # Scale to first half of progress
                progress_callback(current // 2, len(paths))

        loaded, errors = image_loader.load_batch(to_process, load_progress)
        failed = [path for path, _ in errors]

        if loaded:
            # Generate embeddings for loaded images
            def gen_progress(current, total):
                if progress_callback:
                    # Scale to second half of progress
                    base = len(embeddings) + len(to_process) // 2
                    progress_callback(base + current, len(paths))

            new_embeddings = self.generate_batch(loaded, gen_progress)
            embeddings.update(new_embeddings)

        return embeddings, failed

    def get_embedding_dim(self) -> int:
        """Return the dimension of CLIP embeddings (768 for ViT-L/14)."""
        return 768
