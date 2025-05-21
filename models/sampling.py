import torch
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from .search import euclidean_dist
from anomalib.models.components.dimensionality_reduction import SparseRandomProjection

warnings.filterwarnings("ignore")


"""
k-Center Greedy Method.

Returns points that minimizes the maximum distance of any point to a center.
- https://arxiv.org/abs/1708.00489
"""

class KCenterGreedy:
    """k-center-greedy method for coreset selection.

    This class implements the k-center-greedy method to select a coreset from an
    embedding space. The method aims to minimize the maximum distance between any
    point and its nearest center.

    Args:
        embedding (torch.Tensor): Embedding tensor extracted from a CNN.
        sampling_ratio (float): Ratio to determine coreset size from embedding size.

    Attributes:
        embedding (torch.Tensor): Input embedding tensor.
        coreset_size (int): Size of the coreset to be selected.
        model (SparseRandomProjection): Dimensionality reduction model.
        features (torch.Tensor): Transformed features after dimensionality reduction.
        min_distances (torch.Tensor): Minimum distances to cluster centers.
        n_observations (int): Number of observations in the embedding.

    Example:
        >>> import torch
        >>> embedding = torch.randn(219520, 1536)
        >>> sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.001)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """
    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """
        embedding: (num_nodes, dim + 1), where the last dim is hyperedge index.
        sampling_ratio: float in (0,1), ratio of selected nodes.
        """
        self.embedding = embedding  # shape: (N, dim + 1)
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)

        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances to None."""
        self.min_distances = None

    def update_distances(self, cluster_centers) -> None:
        """Update minimum distances given cluster centers.

        Args:
            cluster_centers (list[int]): Indices of cluster centers.
        """
        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index of the next sample based on maximum minimum distance.

        Returns:
            int: Index of the selected sample.

        Raises:
            TypeError: If `self.min_distances` is not a torch.Tensor.
        """
        if isinstance(self.min_distances, torch.Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
            raise TypeError(msg)

        return idx

    def select_coreset_idxs(self, selected_idxs=None):
        """Greedily form a coreset to minimize maximum distance to cluster centers.

        Args:
            selected_idxs (list[int] | None, optional): Indices of pre-selected
                samples. Defaults to None.

        Returns:
            list[int]: Indices of samples selected to minimize distance to cluster
                centers.

        Raises:
            ValueError: If a newly selected index is already in `selected_idxs`.
        """
        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.model.fit(self.embedding[:, :-1])
            self.features = self.model.transform(self.embedding[:, :-1])
            self.reset_distances()
        else:
            self.features = self.embedding[:, :-1].reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: list[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        for _ in range(self.coreset_size):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs=None) -> torch.Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs (list[int] | None, optional): Indices of pre-selected
                samples. Defaults to None.

        Returns:
            torch.Tensor: Selected coreset.

        Example:
            >>> import torch
            >>> embedding = torch.randn(219520, 1536)
            >>> sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.001)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """
        idxs = self.select_coreset_idxs(selected_idxs)
        return self.embedding[idxs]


def sample_nodes(node_bank, num_clusters, sampling_ratio):
    """
    Sample nodes for each hyperedge using K-Center-Greedy algorithm.

    Args:
        node_bank (Tensor): Node bank tensor of shape (N, dim + 1), where the last column is cluster ID.
        num_clusters (int): Total number of clusters (hyperedges).
        sampling_ratio (float): Ratio of total nodes to sample.

    Returns:
        Tensor: Sampled nodes of shape (M, dim + 1), where M ≈ N * sampling_ratio.
    """
    sampled_nodes = []
    total_nodes = node_bank.size(0)
    total_target = max(1, int(total_nodes * sampling_ratio))

    for cid in range(num_clusters):
        mask = node_bank[:, -1] == float(cid)
        cluster_nodes = node_bank[mask]

        if cluster_nodes.size(0) == 0:
            continue

        n_sample = max(1, int(round(cluster_nodes.size(0) / total_nodes * total_target)))

        if cluster_nodes.size(0) == 1 or n_sample == 1:
            feats = cluster_nodes[:, :-1]
            dists = torch.cdist(feats, feats, p=2)
            max_dists = dists.max(dim=1).values
            best_idx = torch.argmin(max_dists).item()
            sampled_nodes.append(cluster_nodes[best_idx].unsqueeze(0))
        else:
            sampler = KCenterGreedy(embedding=cluster_nodes, sampling_ratio=n_sample / cluster_nodes.size(0))
            idxs = sampler.select_coreset_idxs()
            sampled_nodes.append(cluster_nodes[idxs])

    return torch.cat(sampled_nodes, dim=0)


def update_coreset_node(
        total_edges: torch.Tensor,
        edge_coreset: torch.Tensor,
        node_coreset: torch.Tensor
    ) -> torch.Tensor:
    """
    Update the last-column index in node_coreset using a coreset-matching map derived from edge features.

    Args:
        total_edges (Tensor): Tensor of shape (N1, dim + 1), original edge features with indices.
        edge_coreset (Tensor): Tensor of shape (N2, dim + 1), selected coreset edge features with indices.
        node_coreset (Tensor): Tensor of shape (N3, dim + 1), node features with cluster index to be updated.

    Returns:
        Tensor: Updated node_coreset tensor with new last-column indices, shape (N3, dim + 1).
    """
    assert total_edges.size(1) == edge_coreset.size(1) and total_edges.size(1) == node_coreset.size(1)

    total_feats, total_index = total_edges[:, :-1], total_edges[:, -1].long()
    sampled_feats, sampled_index = edge_coreset[:, :-1], edge_coreset[:, -1].long()

    dists = euclidean_dist(total_feats, sampled_feats)

    min_indices = torch.argmin(dists, dim=1)
    matched_sampled_ids = sampled_index[min_indices]
    mapping = torch.stack([total_index, matched_sampled_ids], dim=1)
    mapping_dict = {int(k.item()): int(v.item()) for k, v in mapping}

    node_coreset_new = node_coreset.clone()

    original_node_indices = node_coreset[:, -1].long()
    updated_indices = torch.tensor(
        [mapping_dict.get(int(idx.item()), int(idx.item())) for idx in original_node_indices],
        dtype=torch.float32,
        device=node_coreset.device
    )

    node_coreset_new[:, -1] = updated_indices

    return node_coreset_new
