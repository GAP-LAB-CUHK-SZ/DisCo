# DisCo:Diffusion-based Cross-modal Shape Reconstruction

## Installation
The following steps have been tested on Ubuntu20.04.
- You must have an NVIDIA graphics card with at least 12GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- Install `PyTorch >= 1.12`. We have tested on `torch1.12.1+cu113` and `torch2.0.0+cu118`, but other versions should also work fine.

```sh
# torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

## Data preparation
Download the preprocessed data from <a href="https://pan.baidu.com/s/1X6k82UNG-1hV_FIthnlwcQ?pwd=r7vs">
BaiduYun (code: r7vs)<a/>. (I will update other download methods) Put all the downloaded data under LASA, unzip the align_mat_all.zip mannually. 
You can choose to use the script ./process_scripts/unzip_all_data to unzip all the data in occ_data and other_data by following commands:
```angular2html
cd process_scripts
python unzip_all_data.py --unzip_occ --unzip_other
```
Run the following commands to generate augmented partial point cloud for synthetic dataset and LASA dataset
```angular2html
cd process_scripts
python augment_arkit_partial_point.py --cat arkit_chair arkit_stool ...
python augment_synthetic_partial_point.py --cat 03001627 future_chair ABO_chair future_stool ...
```
Run the following command to extract image features
```angular2html
cd process_scripts
bash dist_extract_vit.sh
```
Finally, run the following command to generate train/val splits, please check ./dataset/taxonomy for the sub-cateory definition:
```angular2html
cd process_scripts
python generate_split_for_arkit --cat arkit_chair arkit_stool ...
python generate_split_for_synthetic_data.py --cat 03001627 future_chair ABO_chair future_stool ...
```

## Evaluation
Download the pretrained weight for each category from <a href="https://pan.baidu.com/s/10liUOaC4CXGn7bN6SQkZsw?pwd=hlf9"> checkpoint.<a/> (code:hlf9). 
Put these folder under LASA/output.<br> The ae folder stores the VAE weight, dm folder stores the diffusion model trained on synthetic data.
finetune_dm folder stores the diffusion model finetuned on LASA dataset.
Run the following commands to evaluate and extract the mesh:
```angular2html
cd evaluation
bash dist_eval.sh
```
The category entries are the sub-category from arkit scenes, please see ./datasets/taxonomy.py about how they are defined.
For example, if you want to evaluate on LASA's chair, category should contain both arkit_chair and arkit_stool. 
make sure the --ae-pth and --dm-pth entry points to the correct checkpoint path. If you are evaluating on LASA,
make sure the --dm-pth points to the finetuned weight in the ./output/finetune_dm folder. The result will be saved
under ./output_result.

## Training
Run the <strong>train_VAE.sh</strong> to train the VAE model. The --category entry in the script specify which category to train on. If you aims to train on one category, just specify one category from <strong> chair, 
cabinet, table, sofa, bed, shelf</strong>. Inputting <strong>all</strong> will train on all categories. Makes sure to download and preprocess all 
the required sub-category data. The sub-category arrangement can be found in ./datasets/taxonomy.py <br>
After finish training the VAE model, run the following commands to pre-extract the VAE features for every object:
```angular2html
cd process_scripts
bash dist_export_triplane_features.sh
```
Then, we can start training the diffusion model on the synthetic dataset by running the <strong>train_diffusion.sh</strong>.<br>
Finally, finetune the diffusion model on LASA dataset by running <strong> finetune_diffusion.sh</strong>. <br><br>

Early stopping is used by mannualy stopping the training by 150 epochs and 500 epochs for training VAE model and diffusion model respetively.
All experiments in the paper are conducted on 8 A100 GPUs with batch size = 22.
