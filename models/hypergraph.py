import os
import math
import time
import torch
import warnings
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")


class BiTFMP(nn.Module):
    """
    Bi-directional Training-Free Message Passing (Bi-TF-MP) module for enhancing node features by cross-hypergraph propagation.

    Args:
        L (int): Number of message passing layers.
        alpha (float): Self-information retention ratio.
        topk (int): Number of top similar nodes used in cross-incidence construction.
        bidirectional (bool): Whether to include reverse hyperedges.
        eps (float): Small constant for numerical stability.
    """
    def __init__(self,
                 L: int = 3,
                 alpha: float = 0.9,
                 topk: int = 1,
                 bidirectional: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        self.L = L
        self.alpha = alpha
        self.topk = topk
        self.bidirectional = bidirectional
        self.eps = eps

    def forward(self, features, adj, total_feats, mem_adj):
        """
        Perform message passing over the joint hypergraph between test samples and memory bank.

        Args:
            features (Tensor): Test node features, shape (B, C, N)
            adj (Tensor): Incidence matrix for test hypergraph, shape (B, N, K)
            total_feats (Tensor): Memory bank node features, shape (A, C)
            mem_adj (Tensor): Incidence matrix for memory hypergraph, shape (A, K)

        Returns:
            X_hat (Tensor): Updated test features after propagation, shape (B, C, N)
            X_E (Tensor): Aggregated edge features from updated nodes, shape (B, K, C)
        """
        B, C, N = features.shape
        A = total_feats.size(0)

        memory_X  = total_feats.unsqueeze(0).expand(B, -1, -1)
        memory_H  = mem_adj.unsqueeze(0).expand(B, -1, -1)

        memory_X  = memory_X.to(features.dtype).to(features.device)
        memory_H  = memory_H.to(features.dtype).to(features.device)

        H_cross = self.build_cross_incidence(features, memory_X)
        H_joint = self.make_joint_H(adj, memory_H, H_cross)

        joint_X = torch.cat([features, memory_X.permute(0, 2, 1)], dim=2)
        joint_X_hat = self.tf_mp(joint_X, H_joint)

        X_hat = joint_X_hat[:, :, :N]

        D_HE = torch.diag_embed(1.0 / torch.clamp(adj.sum(dim=1), min=self.eps))
        D_HV = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(adj.sum(dim=2), min=self.eps)))
        X_E = D_HE @ adj.transpose(1, 2) @ D_HV @ X_hat.transpose(1, 2)

        return X_hat, X_E
    
    def tf_mp(self, X, H):
        """
        Training-Free Message Passing layer using symmetric normalized clique expansion.

        Args:
            X (Tensor): Node features, shape (B, C, N)
            H (Tensor): Incidence matrix, shape (B, N, M)

        Returns:
            X_hat (Tensor): Updated node features, shape (B, C, N)
        """
        bs, C, N = X.shape
        _, _, M = H.shape
        D_E = H.sum(dim=1) + self.eps
        D_E_inv = 1.0 / D_E

        D_E_diag = torch.diag_embed(D_E_inv)

        H_weighted = torch.bmm(H, D_E_diag)
        W = torch.bmm(H_weighted, H.transpose(1, 2))

        I = torch.eye(N, device=X.device).unsqueeze(0).expand(bs, -1, -1)
        W_tilde = W + I

        D = W_tilde.sum(dim=2) + self.eps
        D_inv_sqrt = 1.0 / torch.sqrt(D)
        D_diag = torch.diag_embed(D_inv_sqrt)
        A = torch.bmm(torch.bmm(D_diag, W_tilde), D_diag)

        A_power = torch.eye(N, device=X.device).unsqueeze(0).expand(bs, -1, -1)
        S = self.alpha * A_power.clone()
        for l in range(1, self.L):
            A_power = torch.bmm(A_power, A)
            S += self.alpha * (1 - self.alpha) ** l * A_power

        A_power = torch.bmm(A_power, A)
        S += (1 - self.alpha) ** self.L * A_power

        X_trans = X.transpose(1, 2)
        X_hat = torch.bmm(S, X_trans)

        return X_hat.transpose(1, 2)
    
    def build_cross_incidence(self, X, mem_X):
        """
        Construct cross-hyperedges between test and memory nodes.

        Args:
            X (Tensor): Test node features, shape (B, C, N)
            mem_X (Tensor): Memory node features, shape (B, A, C)

        Returns:
            H (Tensor): Combined incidence matrix with cross-hyperedges, shape (B, N+A, E_cross)
        """
        B, C, N = X.shape
        A = mem_X.size(1)
        device = X.device

        x_n = F.normalize(X, dim=1)
        m_n = F.normalize(mem_X, dim=2)
        m_n = m_n.permute(0, 2, 1)

        sim = torch.bmm(x_n.permute(0, 2, 1), m_n)
        knn = sim.topk(k=self.topk, dim=-1).indices

        E_cross = N + (A if self.bidirectional else 0)
        H = torch.zeros(B, N + A, E_cross, device=device)

        for b in range(B):
            idx = torch.arange(N, device=device)
            H[b, idx, idx] = 1.
            for k in range(self.topk):
                dst = knn[b, :, k] + N
                H[b].scatter_(0,
                    dst.unsqueeze(0),
                    torch.ones_like(dst, dtype=H.dtype).unsqueeze(0))

        if self.bidirectional:
            sim_rev = sim.transpose(1, 2)
            rev = sim_rev.topk(k=self.topk, dim=-1).indices
            base = N
            for b in range(B):
                for a in range(A):
                    e = base + a
                    H[b, N + a, e] = 1.
                    dst = rev[b, a]
                    H[b, dst, e] = 1.

        return H
    
    def make_joint_H(self, adj, mem_adj, H_cross):
        """
        Concatenate test, memory, and cross-hyperedges into a joint incidence matrix.

        Args:
            adj (Tensor): Test incidence matrix, shape (B, N, K)
            mem_adj (Tensor): Memory incidence matrix, shape (B, A, K)
            H_cross (Tensor): Cross incidence matrix, shape (B, N+A, E_cross)

        Returns:
            H_joint (Tensor): Joint incidence matrix, shape (B, N+A, K+K+E_cross)
        """
        B, N, K = adj.shape
        A = mem_adj.size(1)
        N_tot = N + A

        pad_mem = torch.zeros(B, A, K, device=adj.device)
        H_test = torch.cat([adj, pad_mem], dim=1)

        pad_test = torch.zeros(B, N, K, device=adj.device)
        H_mem = torch.cat([pad_test, mem_adj], dim=1)

        return torch.cat([H_test, H_mem, H_cross], dim=2)


def construct_hypergraph(features, mask, num_clusters, threshold=0.9, eps=1e-12, adj=None):
    """
    Construct a fuzzy hypergraph using cosine similarity-based thresholding.

    Args:
        features (Tensor): Input node features, shape (B, C, N)
        mask (Tensor): Binary foreground mask, shape (B, 1, H, W)
        num_clusters (int): Number of hyperedges (clusters)
        threshold (float): Cosine similarity threshold for assignment
        eps (float): Stability term for division
        adj (Tensor, optional): Precomputed adjacency matrix

    Returns:
        hypergraph_adj (Tensor): Hypergraph adjacency matrix, shape (B, N, K)
        hyperedge_feature (Tensor): Aggregated hyperedge features, shape (B, K, C)
    """
    bs, dim, num_patches = features.shape
    device = features.device

    if adj is None:
        hypergraph_adj = torch.zeros((bs, num_patches, num_clusters), device=device)
        mask = mask.reshape(features.shape[0], -1)

        for b in range(bs):
            feat = features[b].T
            valid_mask = mask[b].bool()
            valid_feat = feat[valid_mask]

            valid_feat_cpu = valid_feat.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
            cluster_labels = kmeans.fit_predict(valid_feat_cpu)
            cluster_centroids = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float32)

            normalized_feat = F.normalize(feat, p=2, dim=1)
            normalized_centroids = F.normalize(cluster_centroids, p=2, dim=1)
            cosine_sim = torch.mm(normalized_feat, normalized_centroids.T)
            
            min_val = cosine_sim.min(dim=1, keepdim=True).values
            max_val = cosine_sim.max(dim=1, keepdim=True).values
            denom = (max_val - min_val)
            denom[denom == 0] = 1e-8
            cosine_similarity = (cosine_sim - min_val) / denom
            
            adaptive_assignment = torch.zeros_like(cosine_similarity)

            adaptive_assignment = (cosine_similarity >= threshold).float()
            hypergraph_adj[b] = adaptive_assignment
            hypergraph_adj[b, ~valid_mask, :] = 0

    else:
        hypergraph_adj = adj
    
    D_HE = torch.diag_embed(1.0 / torch.clamp(hypergraph_adj.sum(dim=1), min=eps))
    D_HV = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(hypergraph_adj.sum(dim=2), min=eps)))
    X_E = D_HE @ hypergraph_adj.transpose(1, 2) @ D_HV @ features.transpose(1, 2)

    return hypergraph_adj, X_E

