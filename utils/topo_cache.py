"""
Topological Metrics Cache
=========================

Cache system for expensive topological computations to avoid redundant calculations.
Supports both single graphs (node classification) and multi-graph datasets (graph classification).
"""

import hashlib
import pickle
import os
from pathlib import Path
import torch
import numpy as np
import warnings


class TopologicalCache:
    """
    Cache for topological metrics (TRF, node weights, etc.)

    The cache is stored both in memory and on disk for persistence across runs.
    """

    def __init__(self, cache_dir='./cache/topo_metrics'):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}

    def _compute_graph_hash(self, edge_index, num_nodes, node_features=None, use_features=False):
        """
        Compute a unique hash for a graph based on its structure.

        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes
            node_features: Optional node features (only hashed if use_features=True)
            use_features: Whether to include features in hash

        Returns:
            str: Unique hash for the graph
        """
        # Convert to numpy for consistent hashing
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()

        # Sort edges for consistent hashing
        edges_sorted = np.sort(edge_index, axis=0)
        edges_sorted = edges_sorted[:, np.lexsort((edges_sorted[1], edges_sorted[0]))]

        # Create hash components
        hash_components = [
            str(num_nodes).encode(),
            edges_sorted.tobytes()
        ]

        # Include features if requested (for homophily-based scaling)
        if use_features and node_features is not None:
            if isinstance(node_features, torch.Tensor):
                node_features = node_features.cpu().numpy()
            # Use feature statistics instead of full features for more robust hashing
            feat_stats = np.concatenate([
                node_features.mean(axis=0),
                node_features.std(axis=0),
                node_features.min(axis=0),
                node_features.max(axis=0)
            ])
            hash_components.append(feat_stats.tobytes())

        # Compute hash
        hasher = hashlib.sha256()
        for component in hash_components:
            hasher.update(component)

        return hasher.hexdigest()

    def _get_cache_key(self, graph_hash, metric_type, **kwargs):
        """
        Generate cache key for a specific metric.

        Args:
            graph_hash: Hash of the graph
            metric_type: Type of metric ('trf', 'node_weights')
            **kwargs: Additional parameters (e.g., trf_weights, gradient_scaling_mode)

        Returns:
            str: Cache key
        """
        # Sort kwargs for consistent key generation
        kwargs_str = '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{metric_type}_{graph_hash}_{kwargs_str}"

    def get(self, edge_index, num_nodes, metric_type, node_features=None, use_features=False, **kwargs):
        """
        Retrieve cached metric if available.

        Args:
            edge_index: Edge index tensor
            num_nodes: Number of nodes
            metric_type: Type of metric to retrieve
            node_features: Optional node features
            use_features: Whether features are used in the metric
            **kwargs: Additional parameters for the metric

        Returns:
            Cached value or None if not found
        """
        graph_hash = self._compute_graph_hash(edge_index, num_nodes, node_features, use_features)
        cache_key = self._get_cache_key(graph_hash, metric_type, **kwargs)

        # Check memory cache first
        if cache_key in self.memory_cache:
            print(f"  [Cache HIT - Memory] {metric_type}")
            return self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                # Store in memory cache for faster subsequent access
                self.memory_cache[cache_key] = value
                print(f"  [Cache HIT - Disk] {metric_type}")
                return value
            except Exception as e:
                warnings.warn(f"Failed to load cache file {cache_file}: {e}")
                # Delete corrupted cache file
                cache_file.unlink()

        print(f"  [Cache MISS] {metric_type}")
        return None

    def set(self, edge_index, num_nodes, metric_type, value, node_features=None, use_features=False, **kwargs):
        """
        Store metric in cache.

        Args:
            edge_index: Edge index tensor
            num_nodes: Number of nodes
            metric_type: Type of metric to store
            value: Value to cache
            node_features: Optional node features
            use_features: Whether features are used in the metric
            **kwargs: Additional parameters for the metric
        """
        graph_hash = self._compute_graph_hash(edge_index, num_nodes, node_features, use_features)
        cache_key = self._get_cache_key(graph_hash, metric_type, **kwargs)

        # Store in memory cache
        self.memory_cache[cache_key] = value

        # Store in disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            print(f"  [Cache STORED] {metric_type}")
        except Exception as e:
            warnings.warn(f"Failed to write cache file {cache_file}: {e}")

    def clear(self):
        """Clear all caches (memory and disk)."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                warnings.warn(f"Failed to delete cache file {cache_file}: {e}")
        print("Cache cleared")


# Global cache instance
_global_cache = None

def get_global_cache():
    """Get or create the global topological cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = TopologicalCache()
    return _global_cache
