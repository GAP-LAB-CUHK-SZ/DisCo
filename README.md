# DisCo:Diffusion-based Cross-modal Shape Reconstruction

## Installation
The following steps have been tested on Ubuntu20.04.
- You must have an NVIDIA graphics card with at least 12GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- Install `PyTorch==2.3.0` and `torchvision==0.18.0`.
```sh
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

- Install DisCo:

```sh
pip install -e .
```

## Data preparation
1. **Download and Organize Data**
   - Download the preprocessed data from [BaiduYun (code: r7vs)](https://pan.baidu.com/s/1X6k82UNG-1hV_FIthnlwcQ?pwd=r7vs).
   - After downloading, place all the data under the `LASA` directory.
   - Unzip `align_mat_all.zip` manually.

2. **Unzip Additional Data**
   - You can use the provided script to unzip all data in `occ_data` and `other_data` directories.
   - Run the script to unzip the data:
     ```sh
     python datasets_preprocess/unzip_all_data.py --unzip_occ --unzip_other
     ```

3. **(Optional)Generate Augmented Partial Point Cloud**
   
4. **Extract Image Features**
   - Navigate to the `process_scripts` directory:
     ```sh
     cd process_scripts
     ```
   - Run the script to extract image features:
     ```sh
     bash dist_extract_vit.sh
     ```

5. **Generate Train/Validation Splits**
   - Navigate to the `process_scripts` directory:
     ```sh
     cd process_scripts
     ```
   - For the LASA dataset, run:
     ```sh
     python generate_split_for_arkit.py --cat arkit_chair arkit_stool ...
     ```
   - For the synthetic dataset, run:
     ```sh
     python generate_split_for_synthetic_data.py --cat 03001627 future_chair ABO_chair future_stool ...
     ```

## Training
All experiments are conducted on 8 A100 GPUs with a batch size of 22. Ensure you have access to similar hardware for optimal performance.
1. **Train the VAE Model**
   - Open the script `train_VAE.sh` with your preferred text editor and ensure the `--category` entry specifies the category you wish to train on. Possible options include:
     - Individual categories: `chair`, `cabinet`, `table`, `sofa`, `bed`, `shelf`
     - All categories: `all`
     - (Make sure you have downloaded and preprocessed all necessary sub-category data as outlined in `./datasets/taxonomy.py`)
   - Run the script to start training the VAE model:
     ```sh
      python train.py --gpus 0,1,2,3,4,5,6,7 --data_path ./data/ --train_type vae
     ```
   - Note: Ensure you use early stopping by manually stopping the training at 150 epochs if needed.

2. **Pre-Extract VAE Features**
   Comming Soon

3. **Train the Diffusion Model on Synthetic Dataset**
   Comming Soon

4. **Finetune the Diffusion Model on LASA Dataset**
   Comming Soon
