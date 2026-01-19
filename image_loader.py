"""Unified image loading with HEIC support."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable

from PIL import Image

# Register HEIC opener
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False
    logging.warning("pillow-heif not installed. HEIC support disabled.")

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}


class ImageLoader:
    """Handles discovery and loading of images with multi-format support."""

    def __init__(self, extensions: Optional[set] = None):
        """Initialize loader with supported extensions.

        Args:
            extensions: Set of file extensions to support. Defaults to common formats.
        """
        self.extensions = extensions or SUPPORTED_EXTENSIONS
        if not HEIC_SUPPORTED:
            self.extensions = self.extensions - {'.heic', '.heif'}

    def discover_images(self, directory: Path | str) -> List[Path]:
        """Find all supported image files in directory recursively.

        Args:
            directory: Root directory to search.

        Returns:
            List of paths to discovered images, sorted alphabetically.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        images = []
        for ext in self.extensions:
            # Case-insensitive matching
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))

        # Remove duplicates and sort
        images = sorted(set(images))
        logger.info(f"Discovered {len(images)} images in {directory}")
        return images

    def load_image(self, path: Path | str) -> Optional[Image.Image]:
        """Load a single image and convert to RGB.

        Args:
            path: Path to image file.

        Returns:
            PIL Image in RGB mode, or None if loading failed.
        """
        path = Path(path)
        try:
            with Image.open(path) as img:
                # Convert to RGB (handles RGBA, L, CMYK, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Load into memory before closing
                img.load()
                return img.copy()
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return None

    def load_batch(
        self,
        paths: List[Path],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[Tuple[Path, Image.Image]], List[Tuple[Path, str]]]:
        """Load multiple images, tracking successes and failures.

        Args:
            paths: List of image paths to load.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            Tuple of (successes, errors) where:
            - successes: List of (path, image) tuples
            - errors: List of (path, error_message) tuples
        """
        successes = []
        errors = []
        total = len(paths)

        for i, path in enumerate(paths):
            if progress_callback:
                progress_callback(i + 1, total)

            img = self.load_image(path)
            if img is not None:
                successes.append((path, img))
            else:
                errors.append((path, f"Failed to load"))

        if errors:
            logger.warning(f"Failed to load {len(errors)}/{total} images")

        return successes, errors


def get_image_hash(path: Path | str) -> str:
    """Compute SHA256 hash of image file for caching.

    Args:
        path: Path to image file.

    Returns:
        First 16 characters of SHA256 hex digest.
    """
    import hashlib

    path = Path(path)
    sha256 = hashlib.sha256()

    with open(path, 'rb') as f:
        # Read in chunks for memory efficiency
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    return sha256.hexdigest()[:16]
