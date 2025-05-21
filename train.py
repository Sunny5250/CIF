import os
import torch
import random
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.memory import HyperMemoryBank
from models.feature_extractor import Extractor
from models.hypergraph import construct_hypergraph
from data.mvtec_dataset import MVTecDataset, ALL_CATEGORY_mvtec
from data.eyecandies_dataset import EyeCandDataset, ALL_CATEGORY_eyecandies
from data.mvtec3d_dataset import MVTec3DDataset, ALL_CATEGORY_mvtec3d, NORMALIZE_MEAN, NORMALIZE_STD

warnings.filterwarnings("ignore")


def train(args, category):
    if args.k_shot == -1:
        shot_setting = 'fullshot'
    else:
        shot_setting = f"{args.k_shot}shot"
    
    run_name = f"{args.run_name_head}_{shot_setting}_{args.sampling_ratio}"
        
    embeddings_path = os.path.join(os.path.join(args.checkpoint_path, run_name), category)
    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)

    if args.k_shot > 0 and args.batch_size > args.k_shot:
        args.batch_size = args.k_shot

    num_patches = (args.image_size // args.patch_size) ** 2
    side = int(num_patches ** 0.5)

    feature_extractor = Extractor(
        dataset=args.dataset,
        image_size=args.image_size,
        num_patches=num_patches,
        group_size=args.group_size,
        num_group=args.num_group,
    ).cuda()

    if args.dataset == 'mvtec3d':
        dataset = MVTec3DDataset(
            is_train=True,
            mvtec3d_dir=args.dataset_path + category + "/train/good",
            resize_shape=(args.image_size, args.image_size),
            pc_type=args.pc_type,
            n_fills=args.n_fills,
            bg_thresh=args.bg_thresh,
            k_shot=args.k_shot,
            indices_file=args.indices_file,
            sampling=args.sampling,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )

    elif args.dataset == 'mvtec':
        dataset = MVTecDataset(
            is_train=True,
            mvtec_dir=args.dataset_path + category + "/train/good",
            resize_shape=(args.image_size, args.image_size),
            k_shot=args.k_shot,
            indices_file=args.indices_file,
            sampling=args.sampling,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )
    
    elif args.dataset == 'eyecandies':
        dataset = EyeCandDataset(
            is_train=True,
            eyecand_dir=args.dataset_path + category + "/train/good",
            resize_shape=(args.image_size, args.image_size),
            pc_type=args.pc_type,
            n_fills=args.n_fills,
            bg_thresh=args.bg_thresh,
            k_shot=args.k_shot,
            indices_file=args.indices_file,
            sampling=args.sampling,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    A = len(dataloader.dataset)
    
    rgb_bank = HyperMemoryBank(dim=args.rgb_dim, num_clusters=args.n_clusters)
    if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
        xyz_bank = HyperMemoryBank(dim=args.xyz_dim, num_clusters=args.n_clusters)

    with tqdm(total=A, desc="Extracting feature", unit="sample") as pbar:
        for _, sample_batched in enumerate(dataloader):
            rgb_image = sample_batched["rgb_image"].cuda()
            if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                xyz_image = sample_batched["point_cloud"].cuda()
            elif args.dataset == 'mvtec':
                xyz_image = None
            fore_mask = sample_batched["foreground"].cuda()

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
            fore_mask.shape:                      torch.Size([bs, 1, side, side])
            '''

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

            rgb_features_mem = rgb_features.squeeze().cpu().detach()
            rgb_adj_mem = rgb_adj.squeeze().cpu().detach()
            rgb_hyperedge_mem = rgb_hyperedge.squeeze().cpu().detach()
            rgb_bank.update(rgb_features_mem, rgb_hyperedge_mem, rgb_adj_mem)
            if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
                xyz_features_mem = xyz_features.squeeze().cpu().detach()
                xyz_hyperedge_mem = xyz_hyperedge.squeeze().cpu().detach()
                xyz_bank.update(xyz_features_mem, xyz_hyperedge_mem, rgb_adj_mem)

            pbar.update(rgb_image.shape[0])

    print('Memory bank sampling:')
    size_init = A * num_patches
    sum = min(A * num_patches * args.sampling_ratio, rgb_bank.get_len())
    true_sampling_ratio = sum / rgb_bank.get_len()
    rgb_bank.mem_sampling(true_sampling_ratio)
    rgb_size_final = rgb_bank.get_len()
    rgb_bank.save_memory(embeddings_path, 'rgb_total_feats.pth')
    print(f'RGB initial node size: {size_init}, RGB final node size: {rgb_size_final}')
    if args.dataset == 'mvtec3d' or args.dataset == 'eyecandies':
        xyz_bank.mem_sampling(true_sampling_ratio)
        xyz_size_final = xyz_bank.get_len()
        xyz_bank.save_memory(embeddings_path, 'xyz_total_feats.pth')
        print(f'XYZ initial node size: {size_init}, XYZ final node size: {xyz_size_final}')

    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--run_name_head", type=str, default='SALAD_exp')

    parser.add_argument("--dataset", type=str, default='mvtec3d', choices=['mvtec3d', 'mvtec', 'eyecandies'])
    parser.add_argument("--dataset_path", type=str, default="datasets/mvtec3d/")
    parser.add_argument("--pc_type", type=str, default='pc', choices=['pc', 'depth'])
    parser.add_argument("--k_shot", type=int, default=4) # -1 represents full-shot
    parser.add_argument("--indices_file", type=str, default=None)
    parser.add_argument("--sampling", type=str, default='order', choices=['order', 'file', 'random'])
    parser.add_argument("--checkpoint_path", type=str, default="./saved_models/")

    parser.add_argument('--batch_size', default=1) # keep batch_size at 1, do not change it.
    parser.add_argument('--image_size', default=224)
    parser.add_argument("--rgb_backbone", type=str, default='dinov1', choices=['dinov1'])
    parser.add_argument("--xyz_backbone", type=str, default='ptv1', choices=['ptv1'])
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.9)
    
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--rgb_dim", type=int, default=768)
    parser.add_argument("--xyz_dim", type=int, default=1152)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--num_group", type=int, default=1024)
    parser.add_argument("--lamda", type=int, default=3)
    parser.add_argument("--n_fills", type=int, default=3)
    parser.add_argument("--bg_thresh", type=float, default=7e-3)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument('--sampling_ratio', type=float, default=0.1)
    
    args = parser.parse_args()

    seed = 3407

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if args.k_shot == -1:
        shot_setting = 'fullshot'
    else:
        shot_setting = f"{args.k_shot}shot"
    
    run_name = f"{args.run_name_head}_{shot_setting}"

    with torch.cuda.device(args.gpu_id):
        if args.dataset == 'mvtec3d':
            print('\nTraining Setting')
            print('====================================================')
            print(f'Exp Name Head:     {args.run_name_head}')
            print(f'Shot Setting:      {shot_setting}')
            print(f'Image Size:        {args.image_size}')
            print(f'RGB Backbone:      {args.rgb_backbone}')
            print(f'XYZ Backbone:      {args.xyz_backbone}')
            print(f'XYZ Data Type:     {args.pc_type}')
            print(f'RGB Feature Dim:   {args.rgb_dim}')
            print(f'XYZ Feature Dim:   {args.xyz_dim}')
            print(f'Batch Size:        {args.batch_size}')
            print(f'Sampling Ratio:    {args.sampling_ratio}')
            print(f'Cluster Number:    {args.n_clusters}')
            print(f'Random Seed:       {seed}')
            print('====================================================')
            print('Data path:', args.dataset_path)

            flag = 0

            for classname in ALL_CATEGORY_mvtec3d:
                flag += 1
                print(f'\n[{flag}/{len(ALL_CATEGORY_mvtec3d)}] Train class ' + classname)
                train(args, classname)
        
        elif args.dataset == 'mvtec':
            print('\nTraining Setting')
            print('====================================================')
            print(f'Exp Name Head:     {args.run_name_head}')
            print(f'Shot Setting:      {shot_setting}')
            print(f'Image Size:        {args.image_size}')
            print(f'RGB Backbone:      {args.rgb_backbone}')
            print(f'RGB Feature Dim:   {args.rgb_dim}')
            print(f'Batch Size:        {args.batch_size}')
            print(f'Sampling Ratio:    {args.sampling_ratio}')
            print(f'Cluster Number:    {args.n_clusters}')
            print(f'Random Seed:       {seed}')
            print('====================================================')
            print('Data path:', args.dataset_path)

            flag = 0

            for classname in ALL_CATEGORY_mvtec:
                flag += 1
                print(f'\n[{flag}/{len(ALL_CATEGORY_mvtec)}] Train class ' + classname)
                train(args, classname)
        
        elif args.dataset == 'eyecandies':
            print('\nTraining Setting')
            print('====================================================')
            print(f'Exp Name Head:     {args.run_name_head}')
            print(f'Shot Setting:      {shot_setting}')
            print(f'Image Size:        {args.image_size}')
            print(f'RGB Backbone:      {args.rgb_backbone}')
            print(f'XYZ Backbone:      {args.xyz_backbone}')
            print(f'XYZ Data Type:     {args.pc_type}')
            print(f'RGB Feature Dim:   {args.rgb_dim}')
            print(f'XYZ Feature Dim:   {args.xyz_dim}')
            print(f'Batch Size:        {args.batch_size}')
            print(f'Sampling Ratio:    {args.sampling_ratio}')
            print(f'Cluster Number:    {args.n_clusters}')
            print(f'Random Seed:       {seed}')
            print('====================================================')
            print('Data path:', args.dataset_path)

            flag = 0

            for classname in ALL_CATEGORY_eyecandies:
                flag += 1
                print(f'\n[{flag}/{len(ALL_CATEGORY_eyecandies)}] Train class ' + classname)
                train(args, classname)
        