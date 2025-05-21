import os
import torch
import torch.nn.functional as F
from .sampling import sample_nodes


class HyperMemoryBank:
    """
    A class for managing memory nodes and hyperedges in a hypergraph-based memory bank.

    Args:
        dim (int): Feature dimension of nodes.
        num_clusters (int): Number of hyperedges (clusters) maintained in memory.
    """
    def __init__(self, dim, num_clusters):
        self.dim = dim
        self.num_clusters = num_clusters
        self.node_bank = []
        self.edge_bank = None

    def get_len(self):
        """
        Get the total number of nodes in the memory bank.

        Returns:
            int: Total number of stored nodes.
        """
        return sum([node.size(0) for node in self.node_bank])

    def print_nodes_per_cluster(self):
        """
        Print the number of nodes assigned to each hyperedge.
        """
        all_nodes = torch.cat(self.node_bank, dim=0)
        cluster_ids = all_nodes[:, -1].long()
        counts = torch.bincount(cluster_ids, minlength=self.num_clusters)
        
        for i, count in enumerate(counts.tolist()):
            print(f'hyperedge "{i}": {count} nodes')

    def get_node_bank(self):
        """
        Get the full concatenated memory bank.

        Returns:
            Tensor: All stored nodes, shape (N, dim + 1)
        """
        return torch.cat(self.node_bank, dim=0)

    def update(self, node_feat, edge_feat, adj=None):
        """
        Update the memory bank with new nodes and edge features.

        Args:
            node_feat (Tensor): Node features, shape (dim, num_patches)
            edge_feat (Tensor): Edge features, shape (num_clusters, dim)
            adj (Tensor): Node-hyperedge incidence matrix, shape (num_patches, num_clusters)
        """
        node_feat = node_feat.T

        valid_mask = (adj.max(dim=1).values > 0)
        node_feat = node_feat[valid_mask]
        adj = adj[valid_mask]

        if self.edge_bank is None:
            cluster_ids = torch.argmax(adj, dim=1).unsqueeze(1).float()
            node_with_cluster = torch.cat([node_feat, cluster_ids], dim=1)
            self.node_bank.append(node_with_cluster)
            self.edge_bank = edge_feat.clone()
        else:
            sim = F.cosine_similarity(
                edge_feat.unsqueeze(1),
                self.edge_bank.unsqueeze(0),
                dim=2
            )
            matched_clusters = torch.argmax(sim, dim=1)

            old_cluster_ids = torch.argmax(adj, dim=1)
            mapped_ids = matched_clusters[old_cluster_ids]
            node_with_cluster = torch.cat([node_feat, mapped_ids.unsqueeze(1).float()], dim=1)
            self.node_bank.append(node_with_cluster)

            self.edge_bank = self._compute_edge_features_from_memory()

    def _compute_edge_features_from_memory(self, eps=1e-6):
        """
        Recompute hyperedge features from the current memory bank using weighted aggregation.

        Args:
            eps (float): Small constant to avoid division by zero.

        Returns:
            Tensor: Updated hyperedge features, shape (num_clusters, dim)
        """
        node_bank = torch.cat(self.node_bank, dim=0)
        node_feat = node_bank[:, :-1]
        cluster_ids = node_bank[:, -1].long()

        H = torch.zeros(node_feat.size(0), self.num_clusters)
        H[torch.arange(node_feat.size(0)), cluster_ids] = 1.0

        D_HE = 1.0 / torch.clamp(H.sum(dim=0), min=eps)
        D_HV = 1.0 / torch.sqrt(torch.clamp(H.sum(dim=1), min=eps))

        D_HE_mat = torch.diag(D_HE)
        D_HV_mat = torch.diag(D_HV)

        edge_feat = D_HE_mat @ H.T @ D_HV_mat @ node_feat

        return edge_feat

    def mem_sampling(self, sampling_ratio):
        """
        Sample a subset of nodes from the memory bank using K-Center-Greedy per hyperedge.

        Args:
            sampling_ratio (float): Ratio of total nodes to retain.

        Returns:
            Tensor: Sampled node bank, shape (M, dim + 1)
        """
        all_nodes = torch.cat(self.node_bank, dim=0)
        sampled_nodes = sample_nodes(all_nodes, self.num_clusters, sampling_ratio)
        self.node_bank = [sampled_nodes]

    def save_memory(self, embeddings_path, filename='rgb_total_feats.pth'):
        """
        Save the node memory bank to a file.

        Args:
            embeddings_path (str): Directory path to save the file.
            filename (str): Output file name.
        """
        node_feats = torch.cat(self.node_bank, dim=0)
        save_path = os.path.join(embeddings_path, filename)
        torch.save(node_feats, save_path)
    