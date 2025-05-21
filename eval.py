import os
import time
import torch
import random
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.metrics import AUPRO, IAPS
from torch.utils.data import DataLoader
from models.feature_extractor import Extractor
from torchmetrics import AUROC, AveragePrecision
from models.hypergraph import construct_hypergraph, BiTFMP
from models.search import euclidean_dist, nearest_neighbors, hyper_search

from data.mvtec_dataset import MVTecDataset, ALL_CATEGORY_mvtec
from data.eyecandies_dataset import EyeCandDataset, ALL_CATEGORY_eyecandies
from data.mvtec3d_dataset import MVTec3DDataset, ALL_CATEGORY_mvtec3d, NORMALIZE_MEAN, NORMALIZE_STD

warnings.filterwarnings("ignore")


def evaluate(args, run_name, category, feature_extractor, embeddings_path, num_patches, vis=False):

    seed = 3407

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    with torch.no_grad():

        if args.dataset == 'mvtec3d':
            dataset = MVTec3DDataset(
                is_train=False,
                mvtec3d_dir=args.dataset_path + category + "/test",
                resize_shape=(args.image_size, args.image_size),
                pc_type=args.pc_type,
                n_fills=args.n_fills,
                bg_thresh=args.bg_thresh,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
            )

        elif args.dataset == 'mvtec':
            dataset = MVTecDataset(
                is_train=False,
                mvtec_dir=args.dataset_path + category + "/test",
                resize_shape=(args.image_size, args.image_size),
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
            )
    
        elif args.dataset == 'eyecandies':
            dataset = EyeCandDataset(
                is_train=False,
                eyecand_dir=args.dataset_path + category + "/test",
                resize_shape=(args.image_size, args.image_size),
                pc_type=args.pc_type,
                n_fills=args.n_fills,
                bg_thresh=args.bg_thresh,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        bi_tfmp =  BiTFMP(L=args.layers, alpha=args.alpha, topk=1, bidirectional=True)

        side = int(num_patches ** 0.5)

        seg_IAPS = IAPS().cuda()
        seg_AUPRO = AUPRO().cuda()
        seg_AUROC = AUROC().cuda()
        seg_AP = AveragePrecision().cuda()
        seg_detect_AUROC = AUROC().cuda()

        test_images, test_depths, gts = [], [], []
        rgb_score_list, xyz_score_list, score_list = [], [], []
        A = len(dataloader.dataset)
        time_all = 0.0

        w_l, w_u = 5, 7
        pad_l, pad_u = 2, 3
        weight_l = torch.ones(1, 1, w_l, w_l).cuda()
        weight_l = weight_l/(w_l**2)
        weight_u = torch.ones(1, 1, w_u, w_u).cuda()
        weight_u = weight_u/(w_u**2)

        with tqdm(total=A, desc="Extracting feature", unit="sample") as pbar:
            for _, sample_batched in enumerate(dataloader):
                start = time.time()
                rgb_image = sample_batched["rgb_image"].cuda()
                if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                    xyz_image = sample_batched["point_cloud"].cuda()
                elif args.dataset == 'mvtec':
                    xyz_image = None
                fore_mask = sample_batched["foreground"].cuda()
                mask = sample_batched["mask"].to(torch.int64).cuda()

                fore_mask = F.interpolate(
                    fore_mask.unsqueeze(1),
                    size=side,
                    mode="bilinear",
                    align_corners=False,
                )
                if args.dataset == 'mvtec3d' or args.dataset == 'mvtec':
                    fore_mask = torch.where(
                        fore_mask < 0.5, torch.zeros_like(fore_mask), torch.ones_like(fore_mask)
                    )
                elif args.dataset == 'eyecandies':
                    fore_mask = torch.where(
                        fore_mask > 0, torch.ones_like(fore_mask), torch.zeros_like(fore_mask)
                    )
                '''
                rgb_image.shape:                      torch.Size([bs, 3, 224, 224])
                xyz_image.shape:                      torch.Size([bs, 3, 224, 224])
                fore_mask.shape:                      torch.Size([bs, 1, 28, 28])
                mask.shape:                           torch.Size([bs, 1, 224, 224])
                '''

                rgb_image_copy = rgb_image.clone()
                mask_copy = mask.clone()
                test_images.append(rgb_image_copy.cpu().detach())
                gts.append(mask_copy.cpu().detach())
                if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                    xyz_image_copy = xyz_image.clone()
                    test_depths.append(xyz_image_copy.cpu().detach())

                rgb_features, xyz_features = feature_extractor(rgb_image, xyz_image)
                '''
                rgb_features.shape:                   torch.Size([bs, rgb_dim, num_patches])
                xyz_features.shape:                   torch.Size([bs, xyz_dim, num_patches])
                '''

                rgb_adj, rgb_hyperedge = construct_hypergraph(rgb_features, fore_mask, args.n_clusters)
                if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                    _, xyz_hyperedge = construct_hypergraph(xyz_features, fore_mask, args.n_clusters, adj=rgb_adj)
                '''
                rgb_adj.shape:                        torch.Size([bs, num_patches, n_clusters])
                rgb_hyperedge.shape:                  torch.Size([bs, n_clusters, rgb_dim])
                xyz_hyperedge.shape:                  torch.Size([bs, n_clusters, xyz_dim])
                '''

                rgb_total_feats = torch.load(os.path.join(embeddings_path, 'rgb_total_feats.pth')).cuda()
                if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                    xyz_total_feats = torch.load(os.path.join(embeddings_path, 'xyz_total_feats.pth')).cuda()
                '''
                rgb_total_feats.shape:                torch.Size([(A * num_patches) * sampling_ratio, rgb_dim + 1])
                xyz_total_feats.shape:                torch.Size([(A * num_patches) * sampling_ratio, xyz_dim + 1])
                '''

                rgb_mem_adj, rgb_total_feats_new = build_adj(rgb_total_feats)
                rgb_memory_bank_node = rgb_total_feats_new[:, :-1].permute(1, 0).unsqueeze(0)
                rgb_mem_adj = rgb_mem_adj.unsqueeze(0)
                d_he_mem = torch.diag_embed(1.0 / torch.clamp(rgb_mem_adj.sum(dim=1), min=1e-12))
                d_hv_mem = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(rgb_mem_adj.sum(dim=2), min=1e-12)))
                rgb_memory_bank_edge = d_he_mem @ rgb_mem_adj.transpose(1, 2) @ d_hv_mem @ rgb_memory_bank_node.transpose(1, 2)

                rgb_features_mp, rgb_hyperedge_mp = bi_tfmp(
                    rgb_features,
                    rgb_adj,
                    rgb_total_feats[:, :-1],
                    rgb_mem_adj.squeeze(0)
                )

                rgb_memory_bank_edge = rgb_memory_bank_edge.squeeze(0)
                id_edge = torch.arange(rgb_memory_bank_edge.shape[0], device=rgb_memory_bank_edge.device).unsqueeze(1).float()
                rgb_memory_bank_edge = torch.cat([rgb_memory_bank_edge, id_edge], dim=1)
                rgb_test_feats = rgb_features.permute(0, 2, 1).reshape(-1, args.rgb_dim)
                rgb_test_feats_mp = rgb_features_mp.permute(0, 2, 1).reshape(-1, args.rgb_dim)
                rgb_patch_scores_init, rgb_locations = nearest_neighbors(rgb_test_feats_mp, rgb_total_feats[:, :-1], 1, fore_mask=fore_mask)
                rgb_patch_scores = hyper_search(
                    rgb_features_mp,
                    rgb_adj,
                    rgb_hyperedge_mp,
                    rgb_total_feats,
                    rgb_memory_bank_edge,
                ).reshape(-1)
                rgb_score_mix = rgb_patch_scores * rgb_patch_scores_init
                rgb_score = rgb_score_mix.reshape(rgb_features.shape[0], 1, side, side)
                rgb_score = F.interpolate(
                    rgb_score,
                    size=args.image_size,
                    mode='bilinear',
                    align_corners=True,
                )
                rgb_score_mix = rgb_score_mix.reshape((rgb_features.shape[0], -1))
                rgb_locations = rgb_locations.reshape((rgb_features.shape[0], -1))
                rgb_pred_score = compute_anomaly_score(rgb_score_mix, rgb_locations, rgb_test_feats, rgb_total_feats[:, :-1])
                
                if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                    xyz_mem_adj, xyz_total_feats_new = build_adj(xyz_total_feats)
                    xyz_memory_bank_node = xyz_total_feats_new[:, :-1].permute(1, 0).unsqueeze(0)
                    xyz_mem_adj = xyz_mem_adj.unsqueeze(0)
                    d_he_mem = torch.diag_embed(1.0 / torch.clamp(xyz_mem_adj.sum(dim=1), min=1e-12))
                    d_hv_mem = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(xyz_mem_adj.sum(dim=2), min=1e-12)))
                    xyz_memory_bank_edge = d_he_mem @ xyz_mem_adj.transpose(1, 2) @ d_hv_mem @ xyz_memory_bank_node.transpose(1, 2)

                    xyz_features_mp, xyz_hyperedge_mp = bi_tfmp(
                        xyz_features,
                        rgb_adj,
                        xyz_total_feats[:, :-1],
                        xyz_mem_adj.squeeze(0)
                    )

                    xyz_memory_bank_edge = xyz_memory_bank_edge.squeeze(0)
                    id_edge = torch.arange(xyz_memory_bank_edge.shape[0], device=xyz_memory_bank_edge.device).unsqueeze(1).float()
                    xyz_memory_bank_edge = torch.cat([xyz_memory_bank_edge, id_edge], dim=1)
                    xyz_test_feats = xyz_features.permute(0, 2, 1).reshape(-1, args.xyz_dim)
                    xyz_test_feats_mp = xyz_features_mp.permute(0, 2, 1).reshape(-1, args.xyz_dim)
                    xyz_patch_scores_init, xyz_locations = nearest_neighbors(xyz_test_feats_mp, xyz_total_feats[:, :-1], 1, fore_mask=fore_mask)
                    xyz_patch_scores = hyper_search(
                        xyz_features_mp,
                        rgb_adj,
                        xyz_hyperedge_mp,
                        xyz_total_feats,
                        xyz_memory_bank_edge,
                    ).reshape(-1)
                    xyz_score_mix = xyz_patch_scores * xyz_patch_scores_init
                    xyz_score = xyz_score_mix.reshape(xyz_features.shape[0], 1, side, side)
                    xyz_score = F.interpolate(
                        xyz_score,
                        size=args.image_size,
                        mode='bilinear',
                        align_corners=True,
                    )
                    xyz_score_mix = xyz_score_mix.reshape((xyz_features.shape[0], -1))
                    xyz_locations = xyz_locations.reshape((xyz_features.shape[0], -1))
                    xyz_pred_score = compute_anomaly_score(xyz_score_mix, xyz_locations, xyz_test_feats, xyz_total_feats[:, :-1])

                '''
                rgb_score.shape:                      torch.Size([bs, 1, 224, 224])
                xyz_score.shape:                      torch.Size([bs, 1, 224, 224])
                '''

                score = rgb_score
                if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                    score = rgb_score * xyz_score

                score = score.pow(1.5)
                for _ in range(5):
                    score = torch.nn.functional.conv2d(input=score, padding=pad_l, weight=weight_l)
                for _ in range(3):
                    score = torch.nn.functional.conv2d(input=score, padding=pad_u, weight=weight_u)
                
                end = time.time()

                time_all += (end - start)

                rgb_score_list.append(rgb_score.cpu())
                if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                    xyz_score_list.append(xyz_score.cpu())
                score_copy = score.clone()
                score_list.append(score_copy.cpu())

                if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                    score_sample = rgb_pred_score * xyz_pred_score
                elif args.dataset == 'mvtec':
                    score_sample = rgb_pred_score
                mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]

                seg_detect_AUROC.update(score_sample, mask_sample)
                seg_AUROC.update(score.flatten(), mask.flatten())
                seg_AUPRO.update(score, mask)
                seg_IAPS.update(score, mask)
                seg_AP.update(score.flatten(), mask.flatten())
                
                pbar.update(rgb_image.shape[0])

        fps = A / time_all

        iap_seg, iap90_seg = seg_IAPS.compute()
        auc_detect, auc_seg, aupro_seg, ap_seg = (
            seg_detect_AUROC.compute(),
            seg_AUROC.compute(),
            seg_AUPRO.compute(),
            seg_AP.compute(),
        )

        print("================================")
        print("FPS:", round(float(fps), 2))
        print("I-AUROC:", round(float(auc_detect), 4))
        print("P-AUROC:", round(float(auc_seg), 4))
        print("AUPRO:", round(float(aupro_seg), 4))
        print("I-AP:", round(float(iap_seg), 4))
        print("P-AP:", round(float(ap_seg), 4))
        print()

        file_name = os.path.join(args.result_path, f'{run_name}_results.txt')
        with open(file_name, 'a') as file:
            file.write(f"Test class {category}\n")
            file.write("================================\n")
            file.write(f"FPS: {round(float(fps), 2)}\n")
            file.write(f"I-AUROC: {round(float(auc_detect), 4)}\n")
            file.write(f"P-AUROC: {round(float(auc_seg), 4)}\n")
            file.write(f"AUPRO: {round(float(aupro_seg), 4)}\n")
            file.write(f"I-AP: {round(float(iap_seg), 4)}\n")
            file.write(f"P-AP: {round(float(ap_seg), 4)}\n\n")

        seg_detect_AUROC.reset()
        seg_AUROC.reset()
        seg_AUPRO.reset()
        seg_IAPS.reset()
        seg_AP.reset()

        test_images = torch.cat(test_images, dim=0).numpy()
        gts = torch.cat(gts, dim=0).numpy()
        rgb_scores = torch.cat(rgb_score_list, dim=0).numpy()
        if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
            test_depths = torch.cat(test_depths, dim=0).numpy()
            xyz_scores = torch.cat(xyz_score_list, dim=0).numpy()
        elif args.dataset == 'mvtec':
            test_depths = None
            xyz_scores = None
        scores = torch.cat(score_list, dim=0).numpy()

        if vis:
            plot_fig(run_name, test_images, test_depths, rgb_scores, xyz_scores, scores, gts, category)

    return auc_detect, auc_seg, aupro_seg


def test(args, category):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    run_name = args.base_model_name

    num_patches = (args.image_size // args.patch_size) ** 2

    feature_extractor = Extractor(
        dataset=args.dataset,
        image_size=args.image_size,
        num_patches=num_patches,
        group_size=args.group_size,
        num_group=args.num_group,
    ).cuda()

    embeddings_path = os.path.join(os.path.join(args.checkpoint_path, run_name), category)
    
    assert os.path.exists(os.path.join(embeddings_path, "rgb_total_feats.pth"))
    if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':    
        assert os.path.exists(os.path.join(embeddings_path, "xyz_total_feats.pth"))

    auc_detect, auc_seg, aupro_seg = evaluate(args, run_name, category, feature_extractor, embeddings_path, num_patches, args.vis)

    return auc_detect, auc_seg, aupro_seg


def compute_anomaly_score(patch_scores, locations, embedding, memory_bank):
    """Compute Image-Level Anomaly Score.

    Args:
        patch_scores (torch.Tensor): Patch-level anomaly scores
        locations: Memory bank locations of the nearest neighbor for each patch location
        embedding: The feature embeddings that generated the patch scores
        memory_bank: The feature embeddings in memory bank

    Returns:
        Tensor: Image-level anomaly scores
    """
    if args.n_neighbors == 1:
        return patch_scores.amax(1)

    batch_size, num_patches = patch_scores.shape
    max_patches = torch.argmax(patch_scores, dim=1)
    max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
    score = patch_scores[torch.arange(batch_size), max_patches]
    nn_index = locations[torch.arange(batch_size), max_patches]
    nn_sample = memory_bank[nn_index, :]
    memory_bank_effective_size = memory_bank.shape[0]
    _, support_samples = nearest_neighbors(
        nn_sample,
        memory_bank,
        n_neighbors=min(args.n_neighbors, memory_bank_effective_size),
        patch=False
    )
    distances = euclidean_dist(max_patches_features.unsqueeze(1), memory_bank[support_samples])
    weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]

    return weights * score


def build_adj(X):
    """
    Construct a binary incidence matrix from node features with cluster labels.

    Args:
        X (Tensor): Input node features with cluster labels, shape (N, D+1). 
                    The last column contains cluster (hyperedge) indices.

    Returns:
        adj (Tensor): Incidence matrix of shape (N, E), where E is the number of unique hyperedges.
        X_new (Tensor): Updated node features with remapped hyperedge indices, shape (N, D+1)
    """
    features = X[:, :-1]
    edge_indices = X[:, -1].long()

    unique_edges, new_edge_indices = torch.unique(edge_indices, return_inverse=True)

    num_nodes = X.shape[0]
    num_edges = unique_edges.shape[0]
    adj = torch.zeros((num_nodes, num_edges), dtype=torch.float32, device=X.device)
    adj[torch.arange(num_nodes), new_edge_indices] = 1.0

    X_new = torch.cat([features, new_edge_indices.unsqueeze(1).float()], dim=1)

    return adj, X_new


def plot_fig(run_name, test_images, test_depths, rgb_scores, xyz_scores, scores, gts, category):
    for i in range(test_images.shape[0]):
        image = test_images[i]
        gt_mask = gts[i].transpose(1, 2, 0).squeeze()
        img = de_normalization(image)
        rgb_score = rgb_scores[i].transpose(1, 2, 0).squeeze()
        score = scores[i].transpose(1, 2, 0).squeeze()
        rgb_heat_map = rgb_score * 255
        heat_map = score * 255

        if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
            depth = test_depths[i][0]
            xyz_score = xyz_scores[i].transpose(1, 2, 0).squeeze()
            xyz_heat_map = xyz_score * 255

            fig = plt.figure()
            ax0 = fig.add_subplot(231)
            ax0.axis('off')
            ax0.imshow(img)
            ax0.title.set_text('RGB Image')

            ax1 = fig.add_subplot(232)
            ax1.axis('off')
            ax1.imshow(depth)
            ax1.title.set_text('3D Depth')

            ax2 = fig.add_subplot(233)
            ax2.axis('off')
            ax2.imshow(gt_mask, cmap='gray')
            ax2.title.set_text('GroundTruth')

            ax3 = fig.add_subplot(234)
            ax3.axis('off')
            ax3.imshow(img, cmap='gray', interpolation='none')
            ax3.imshow(rgb_heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax3.title.set_text('RGB Heatmap')

            ax4 = fig.add_subplot(235)
            ax4.axis('off')
            ax4.imshow(img, cmap='gray', interpolation='none')
            ax4.imshow(xyz_heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax4.title.set_text('3D Heatmap')

            ax5 = fig.add_subplot(236)
            ax5.axis('off')
            ax5.imshow(img, cmap='gray', interpolation='none')
            ax5.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax5.title.set_text('Heatmap')
        
        elif args.dataset == 'mvtec':
            fig = plt.figure()
            ax0 = fig.add_subplot(131)
            ax0.axis('off')
            ax0.imshow(img)
            ax0.title.set_text('RGB Image')

            ax1 = fig.add_subplot(132)
            ax1.axis('off')
            ax1.imshow(gt_mask, cmap='gray')
            ax1.title.set_text('GroundTruth')

            ax2 = fig.add_subplot(133)
            ax2.axis('off')
            ax2.imshow(img, cmap='gray', interpolation='none')
            ax2.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax2.title.set_text('Heatmap')

        fig.tight_layout()
        if not os.path.exists(os.path.join(os.path.join(args.plot_path, run_name), category)):
            os.makedirs(os.path.join(os.path.join(args.plot_path, run_name), category))
        fig.savefig(os.path.join(os.path.join(os.path.join(args.plot_path, run_name), category), f"{i}.png"), dpi=100)
        plt.close()

        plt.imshow(img, cmap='gray', interpolation='none')
        plt.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        plt.axis('off')
        plt.savefig(os.path.join(os.path.join(os.path.join(args.plot_path, run_name), category), f"{i}_ano.png"), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()


def de_normalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--base_model_name", type=str, default="CFMAD_experience_1500")

    parser.add_argument("--dataset", type=str, default='mvtec3d', choices=['mvtec3d', 'mvtec', 'eyecandies'])
    parser.add_argument("--dataset_path", type=str, default="datasets/mvtec3d/")
    parser.add_argument("--pc_type", type=str, default='pc', choices=['pc', 'depth'])
    parser.add_argument("--checkpoint_path", type=str, default="./saved_models/")
    parser.add_argument("--plot_path", type=str, default="./visualization/")
    parser.add_argument("--result_path", type=str, default="./results/")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--rgb_backbone", type=str, default='dinov1', choices=['dinov1'])
    parser.add_argument("--xyz_backbone", type=str, default='ptv1', choices=['ptv1'])
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--rgb_dim", type=int, default=768)
    parser.add_argument("--xyz_dim", type=int, default=1152)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--num_group", type=int, default=1024)
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument("--n_fills", type=int, default=3)
    parser.add_argument("--bg_thresh", type=int, default=7e-3)
    parser.add_argument('--sampling_ratio', type=float, default=0.1)
    parser.add_argument("--vis", action='store_true', default=False)

    args = parser.parse_args()

    if args.dataset == 'mvtec3d':
        ALL_CATEGORY = ALL_CATEGORY_mvtec3d
    elif args.dataset == 'mvtec':
        ALL_CATEGORY = ALL_CATEGORY_mvtec
    elif args.dataset == 'eyecandies':
        ALL_CATEGORY = ALL_CATEGORY_eyecandies
        
    print('\nData path:', args.dataset_path)

    with torch.cuda.device(args.gpu_id):
        flag = 0
        auc_detect_list = []
        auc_seg_list = []
        aupro_seg_list = []
        for classname in ALL_CATEGORY:
            flag += 1
            print(f'\n[{flag}/{len(ALL_CATEGORY)}] Test class ' + classname)
            auc_detect, auc_seg, aupro_seg = test(args, classname)
            auc_detect_list.append(auc_detect.cpu())
            auc_seg_list.append(auc_seg.cpu())
            aupro_seg_list.append(aupro_seg.cpu())
        
        auc_detect_mean = sum(auc_detect_list) / len(auc_detect_list)
        auc_seg_mean = sum(auc_seg_list) / len(auc_seg_list)
        aupro_seg_mean = sum(aupro_seg_list) / len(aupro_seg_list)

        print()
        print("Mean")
        print("================================")
        print("I-AUROC:", round(float(auc_detect_mean), 4))
        print("P-AUROC:", round(float(auc_seg_mean), 4))
        print("AUPRO:", round(float(aupro_seg_mean), 4))
        print()

        file_name = os.path.join(args.result_path, f'{args.base_model_name}_results.txt')
        with open(file_name, 'a') as file:
            file.write(f"Mean L {args.layers} a {args.alpha}\n")
            file.write("================================\n")
            file.write(f"I-AUROC: {round(float(auc_detect_mean), 4)}\n")
            file.write(f"P-AUROC: {round(float(auc_seg_mean), 4)}\n")
            file.write(f"AUPRO: {round(float(aupro_seg_mean), 4)}\n\n")
