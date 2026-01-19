"""Composition analysis using Ollama vision models."""

import base64
import io
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

import requests
from PIL import Image

from image_loader import ImageLoader

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llava:13b"  # or llava:7b for faster but less accurate
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_SCORE = 5.0  # fallback on error

COMPOSITION_PROMPT = """Analyze this photograph's composition quality. Evaluate these aspects:

1. Rule of thirds - Is the subject well-positioned?
2. Leading lines - Do lines guide the eye effectively?
3. Subject isolation - Is the main subject clear and distinct?
4. Color harmony - Do colors work well together?
5. Depth and layering - Is there visual depth?
6. Visual storytelling - Does the image convey a mood or story?

Based on your analysis, provide a single overall composition score from 1 to 10, where:
- 1-3: Poor composition with significant issues
- 4-5: Average composition, nothing special
- 6-7: Good composition with effective techniques
- 8-10: Excellent composition, professional quality

Respond with ONLY a number between 1 and 10, nothing else."""


class CompositionAnalyzer:
    """Analyze image composition using Ollama vision models."""

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        rate_limit_delay: float = 0.5
    ):
        """Initialize composition analyzer.

        Args:
            ollama_url: Base URL for Ollama API.
            model: Vision model to use (must support images).
            timeout: Request timeout in seconds.
            rate_limit_delay: Delay between API calls in seconds.
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._available = None

    def check_availability(self) -> bool:
        """Check if Ollama is available and has the required model.

        Returns:
            True if Ollama is ready, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            # Check Ollama is running
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]

            # Check for exact match or partial match (e.g., "llava:13b" matches "llava:13b-v1.6")
            base_model = self.model.split(':')[0]
            self._available = any(
                base_model in name for name in model_names
            )

            if not self._available:
                logger.warning(f"Model {self.model} not found. Available: {model_names}")
            else:
                logger.info(f"Ollama available with model {self.model}")

            return self._available

        except requests.RequestException as e:
            logger.warning(f"Ollama not available: {e}")
            self._available = False
            return False

    def _encode_image(self, image: Image.Image, max_size: int = 1024) -> str:
        """Encode image to base64 for API.

        Args:
            image: PIL Image to encode.
            max_size: Maximum dimension (resizes if larger).

        Returns:
            Base64-encoded JPEG string.
        """
        # Resize if needed to reduce API payload
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to JPEG bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _parse_score(self, response: str) -> float:
        """Extract numeric score from LLM response.

        Args:
            response: Raw text response from LLM.

        Returns:
            Score between 1 and 10, or default on parse failure.
        """
        # Try to find a number in the response
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)

        for num_str in numbers:
            try:
                score = float(num_str)
                if 1 <= score <= 10:
                    return score
            except ValueError:
                continue

        logger.warning(f"Could not parse score from response: {response[:100]}")
        return DEFAULT_SCORE

    def analyze_image(self, image: Image.Image) -> float:
        """Analyze a single image's composition.

        Args:
            image: PIL Image to analyze.

        Returns:
            Composition score from 1 to 10.
        """
        if not self.check_availability():
            return DEFAULT_SCORE

        try:
            encoded = self._encode_image(image)

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": COMPOSITION_PROMPT,
                    "images": [encoded],
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            text_response = result.get('response', '')

            return self._parse_score(text_response)

        except requests.Timeout:
            logger.warning("Ollama request timed out")
            return DEFAULT_SCORE
        except requests.RequestException as e:
            logger.warning(f"Ollama request failed: {e}")
            return DEFAULT_SCORE

    def analyze_batch(
        self,
        paths: List[Path],
        image_loader: Optional[ImageLoader] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[Path, float]:
        """Analyze composition for multiple images.

        Args:
            paths: List of image paths to analyze.
            image_loader: ImageLoader instance. Creates one if None.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Dictionary mapping paths to composition scores.
        """
        if not self.check_availability():
            logger.warning("Ollama not available, using default scores")
            return {p: DEFAULT_SCORE for p in paths}

        if image_loader is None:
            image_loader = ImageLoader()

        results = {}
        total = len(paths)

        for i, path in enumerate(paths):
            # Load image
            image = image_loader.load_image(path)
            if image is None:
                results[path] = DEFAULT_SCORE
                continue

            # Analyze
            score = self.analyze_image(image)
            results[path] = score

            if progress_callback:
                progress_callback(i + 1, total)

            # Rate limiting
            if i < total - 1:
                time.sleep(self.rate_limit_delay)

        return results

    def analyze_top_candidates(
        self,
        paths: List[Path],
        aesthetic_scores: Dict[Path, float],
        num_candidates: int = 800,
        image_loader: Optional[ImageLoader] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[Path, float]:
        """Analyze only top aesthetic candidates for efficiency.

        Args:
            paths: List of all image paths.
            aesthetic_scores: Pre-computed aesthetic scores.
            num_candidates: Number of top candidates to analyze.
            image_loader: ImageLoader instance.
            progress_callback: Progress callback.

        Returns:
            Dictionary mapping analyzed paths to composition scores.
            Non-analyzed paths are not included.
        """
        # Sort by aesthetic score and take top N
        scored_paths = [(p, aesthetic_scores.get(p, 0)) for p in paths]
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        top_paths = [p for p, _ in scored_paths[:num_candidates]]

        logger.info(f"Analyzing composition for top {len(top_paths)} candidates")

        return self.analyze_batch(top_paths, image_loader, progress_callback)
