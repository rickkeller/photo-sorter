"""Image deduplication using cosine similarity and Union-Find clustering."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class DuplicateCluster:
    """A cluster of similar images."""
    cluster_id: int
    representative: Path  # Best image in cluster
    members: List[Path] = field(default_factory=list)  # All images including representative

    @property
    def duplicates(self) -> List[Path]:
        """Return non-representative members."""
        return [p for p in self.members if p != self.representative]


class UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_clusters(self) -> Dict[int, List[int]]:
        """Return mapping of root -> member indices."""
        clusters: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters


class ImageDeduplicator:
    """Find and cluster duplicate/similar images using CLIP embeddings."""

    def __init__(self, similarity_threshold: float = 0.92):
        """Initialize deduplicator.

        Args:
            similarity_threshold: Minimum cosine similarity to consider duplicates.
                                 0.92 is good for near-duplicates.
        """
        self.similarity_threshold = similarity_threshold

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Array of shape (N, embedding_dim).

        Returns:
            Similarity matrix of shape (N, N).
        """
        return cosine_similarity(embeddings)

    def find_duplicates(
        self,
        paths: List[Path],
        embeddings: Dict[Path, np.ndarray],
        aesthetic_scores: Optional[Dict[Path, float]] = None
    ) -> tuple[List[Path], List[DuplicateCluster]]:
        """Find duplicate clusters and select representatives.

        Args:
            paths: List of image paths in order.
            embeddings: Dictionary mapping paths to embedding arrays.
            aesthetic_scores: Optional scores for representative selection.
                            Higher score = better. If None, uses first image.

        Returns:
            Tuple of:
            - unique_images: List of representative paths (one per cluster)
            - duplicate_clusters: List of DuplicateCluster objects
        """
        # Filter to paths with embeddings
        valid_paths = [p for p in paths if p in embeddings]
        if len(valid_paths) < len(paths):
            logger.warning(f"Skipping {len(paths) - len(valid_paths)} images "
                         f"without embeddings")

        if not valid_paths:
            return [], []

        n = len(valid_paths)
        logger.info(f"Computing similarity for {n} images...")

        # Build embedding matrix
        embedding_matrix = np.stack([embeddings[p] for p in valid_paths])

        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(embedding_matrix)

        # Cluster using Union-Find
        uf = UnionFind(n)
        pairs_found = 0

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.similarity_threshold:
                    uf.union(i, j)
                    pairs_found += 1

        logger.info(f"Found {pairs_found} similar pairs above threshold "
                   f"{self.similarity_threshold}")

        # Extract clusters
        index_clusters = uf.get_clusters()

        # Convert to paths and select representatives
        unique_images = []
        duplicate_clusters = []
        cluster_id = 0

        for root, member_indices in index_clusters.items():
            member_paths = [valid_paths[i] for i in member_indices]

            # Select representative (highest aesthetic score, or first)
            if aesthetic_scores:
                representative = max(
                    member_paths,
                    key=lambda p: aesthetic_scores.get(p, 0.0)
                )
            else:
                # Default to first (often earliest by filename)
                representative = member_paths[0]

            unique_images.append(representative)

            if len(member_paths) > 1:
                duplicate_clusters.append(DuplicateCluster(
                    cluster_id=cluster_id,
                    representative=representative,
                    members=member_paths
                ))
                cluster_id += 1

        # Sort unique images by original order
        path_order = {p: i for i, p in enumerate(valid_paths)}
        unique_images.sort(key=lambda p: path_order.get(p, float('inf')))

        logger.info(f"Reduced {n} images to {len(unique_images)} unique "
                   f"({len(duplicate_clusters)} duplicate clusters)")

        return unique_images, duplicate_clusters

    def get_cluster_for_path(
        self,
        path: Path,
        clusters: List[DuplicateCluster]
    ) -> Optional[int]:
        """Find which cluster a path belongs to.

        Args:
            path: Image path to look up.
            clusters: List of duplicate clusters.

        Returns:
            Cluster ID if found, None if image is not in any cluster.
        """
        for cluster in clusters:
            if path in cluster.members:
                return cluster.cluster_id
        return None
