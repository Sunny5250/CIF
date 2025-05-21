import os
import torch
from typing import Tuple
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate pair-wise distance between row vectors in x and those in y.

    Replaces torch cdist with p=2, as cdist is not properly exported to onnx and openvino format.
    Resulting matrix is indexed by x vectors in rows and y vectors in columns.

    Args:
        x: input tensor 1
        y: input tensor 2

    Returns:
        Matrix of distances between row vectors in x and y.
    """
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
    # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
    res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)

    return res.clamp_min_(0).sqrt_()


def nearest_neighbors(
        embedding: torch.Tensor,
        memory_bank: torch.Tensor,
        n_neighbors: int,
        patch: bool = True,
        fore_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest Neighbours using brute force method and euclidean norm.

    Args:
        embedding (torch.Tensor): Features to compare the distance with the memory bank.
        memory_bank (torch.Tensor): Memory bank features.
        fore_mask (torch.Tensor): Foreground mask to filter out background patches.
        n_neighbors (int): Number of neighbors to look at.
        patch (bool): Whether to get patch scores or not.

    Returns:
        Tensor: Patch scores.
        Tensor: Locations of the nearest neighbor(s).
    """
    N = embedding.size(0)
    device = embedding.device

    if patch:
        if fore_mask is None:
            raise ValueError("fore_mask must be provided when patch=True")
        
        if n_neighbors == 1:
            patch_scores = torch.zeros(N, device=device)
            locations = torch.zeros(N, dtype=torch.long, device=device)
        else:
            patch_scores = torch.zeros(N, n_neighbors, device=device)
            locations = torch.zeros(N, n_neighbors, dtype=torch.long, device=device)

        fore_mask = fore_mask.reshape(-1).bool()
        fg_idx = torch.nonzero(fore_mask, as_tuple=True)[0]
        distances_fg = euclidean_dist(embedding[fg_idx], memory_bank)
        if n_neighbors == 1:
            scores_fg, locs_fg = distances_fg.min(1)
        else:
            scores_fg, locs_fg = distances_fg.topk(k=n_neighbors, largest=False, dim=1)

        patch_scores[fg_idx] = scores_fg
        locations[fg_idx]    = locs_fg

    else:
        distances = euclidean_dist(embedding, memory_bank)
        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)

    return patch_scores, locations


def hyper_search(
        features_test: torch.Tensor,
        adj: torch.Tensor,
        hyperedge: torch.Tensor,
        total_feats: torch.Tensor,
        total_edges: torch.Tensor,
        topk: int = 3,
    ) -> torch.Tensor:
    """
    Perform node-level anomaly detection by comparing test hyperedges to memory hyperedges.

    Args:
        features_test (Tensor): Test node features, shape (B, D, N)
        adj (Tensor): Node-hyperedge incidence matrix, shape (B, N, K)
        hyperedge (Tensor): Test hyperedge features, shape (B, K, D)
        total_feats (Tensor): Memory node bank, shape (T, D+1), last column is hyperedge index
        total_edges (Tensor): Memory edge bank, shape (T_e, D+1), last column is edge ID
        topk (int): Number of most similar memory edges to retrieve

    Returns:
        Tensor: Anomaly scores for each node, shape (B, N)
    """
    bs, dim, num_patches = features_test.shape
    _, n_clusters, _ = hyperedge.shape

    total_adjs = total_feats[:, -1:]
    
    anomaly_scores = torch.zeros(bs, num_patches, device=features_test.device)

    for b in range(bs):
        sim_matrix  = torch.matmul(F.normalize(hyperedge[b], dim=1), F.normalize(total_edges[:, :-1], dim=1).T)
        sim_matrix = (sim_matrix + 1) / 2

        _, topk_indices = torch.topk(sim_matrix, min(topk, total_edges.shape[0]), dim=1)
        real_indices = total_edges[:, -1]
        real_topk_indices = real_indices[topk_indices]

        node_feats = features_test[b].permute(1, 0).contiguous()
        node_adjs = adj[b]
        node_edge_ids = torch.argmax(node_adjs, dim=-1, keepdim=True)
        no_connection_mask = (node_adjs.sum(dim=-1, keepdim=True) == 0)
        node_edge_ids = torch.where(no_connection_mask, torch.full_like(node_edge_ids, -1, device=features_test.device), node_edge_ids).squeeze(1)

        for eid in range(n_clusters):
            mask_e1 = (node_edge_ids == eid)
            N1 = node_feats[mask_e1]

            e2_indices = real_topk_indices[eid, :]
            mask_e2_all = torch.isin(total_adjs.squeeze(1), e2_indices)
            N2 = total_feats[mask_e2_all, :-1]

            if N1.shape[0] > 0 and N2.shape[0] > 0:
                dist = euclidean_dist(N1, N2)
                min_dists, _ = dist.min(dim=1)
                anomaly_scores[b, mask_e1] = min_dists
            else:
                anomaly_scores[b, mask_e1] = 1e6

    return anomaly_scores
