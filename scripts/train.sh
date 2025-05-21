export NO_ALBUMENTATIONS_UPDATE=1

###########################################################################################################

# MVTec AD-3D

python train.py --gpu_id 0 --run_name_head "CIF_mvtec3d" --k_shot 1 --sampling_ratio 0.1 --n_clusters 4

python train.py --gpu_id 0 --run_name_head "CIF_mvtec3d" --k_shot 2 --sampling_ratio 0.1 --n_clusters 4

python train.py --gpu_id 0 --run_name_head "CIF_mvtec3d" --k_shot 4 --sampling_ratio 0.1 --n_clusters 4

###########################################################################################################

# Eyecandies

python train.py --gpu_id 0 --run_name_head "CIF_eyecandies" --k_shot 1 --sampling_ratio 0.1 --n_clusters 8 --dataset 'eyecandies' --dataset_path '/mnt/share200/cs24-linyx/datasets/eyecandies_preprocessed/'

python train.py --gpu_id 0 --run_name_head "CIF_eyecandies" --k_shot 2 --sampling_ratio 0.1 --n_clusters 8 --dataset 'eyecandies' --dataset_path '/mnt/share200/cs24-linyx/datasets/eyecandies_preprocessed/'

python train.py --gpu_id 0 --run_name_head "CIF_eyecandies" --k_shot 4 --sampling_ratio 0.1 --n_clusters 8 --dataset 'eyecandies' --dataset_path '/mnt/share200/cs24-linyx/datasets/eyecandies_preprocessed/'

###########################################################################################################
