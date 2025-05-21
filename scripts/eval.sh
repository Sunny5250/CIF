export NO_ALBUMENTATIONS_UPDATE=1

###########################################################################################################

# MVTec AD-3D

python eval.py --gpu_id 0 --base_model_name "CIF_mvtec3d_1shot_0.1" --layer 1 --alpha 0.9 --n_clusters 4 --vis

python eval.py --gpu_id 0 --base_model_name "CIF_mvtec3d_2shot_0.1" --layer 1 --alpha 0.9 --n_clusters 4 --vis

python eval.py --gpu_id 0 --base_model_name "CIF_mvtec3d_4shot_0.1" --layer 1 --alpha 0.9 --n_clusters 4 --vis

###########################################################################################################

# Eyecandies

python eval.py --gpu_id 0 --base_model_name "CIF_eyecandies_1shot_0.1" --layer 1 --alpha 0.9 --n_clusters 8 --dataset 'eyecandies' --dataset_path '/mnt/share200/cs24-linyx/datasets/eyecandies_preprocessed/' --vis

python eval.py --gpu_id 0 --base_model_name "CIF_eyecandies_2shot_0.1" --layer 1 --alpha 0.9 --n_clusters 8 --dataset 'eyecandies' --dataset_path '/mnt/share200/cs24-linyx/datasets/eyecandies_preprocessed/' --vis

python eval.py --gpu_id 0 --base_model_name "CIF_eyecandies_4shot_0.1" --layer 1 --alpha 0.9 --n_clusters 8 --dataset 'eyecandies' --dataset_path '/mnt/share200/cs24-linyx/datasets/eyecandies_preprocessed/' --vis

###########################################################################################################
