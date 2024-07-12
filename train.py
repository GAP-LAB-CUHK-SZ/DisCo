import os
import subprocess
import argparse

def run_training(script, configs, output_dir, log_dir, num_workers, batch_size, epochs, warmup_epochs, 
                 dist_eval, category, data_path, replica, acc_iter=2, clip_grad=None, ae_path=None, port=15000, gpus="0,1,2,3,4,5,6,7"):

    cmd = [
        'CUDA_VISIBLE_DEVICES=' + gpus, 'torchrun',
        '--master_port', str(port),
        '--nproc_per_node=8',
        script,
        '--configs', configs,
        '--accum_iter', str(acc_iter),
        '--output_dir', output_dir,
        '--log_dir', log_dir,
        '--num_workers', str(num_workers),
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--warmup_epochs', str(warmup_epochs),
        '--category', category,
        '--data-pth', data_path,
        '--replica', str(replica)
    ]

    if dist_eval:
        cmd.append('--dist_eval')
    if clip_grad is not None:
        cmd.extend(['--clip_grad', str(clip_grad)])
    if ae_path is not None:
        cmd.extend(['--ae-pth', ae_path])

    subprocess.run(' '.join(cmd), shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Triplane Models')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7", help='Visible GPUs, default: "0,1,2,3,4,5,6,7"')
    parser.add_argument('--train_type', type=str, choices=['vae', 'diffusion', 'both'], default='both', help='Training type to run: vae, diffusion, or both')

    args = parser.parse_args()
    
    base_dir = "../output"
    data_path = args.data_path
    gpus = args.gpus
    train_type = args.train_type

    if train_type in ['vae', 'both']:
        # Train Triplane VAE
        run_training(
            script="train_triplane_vae.py",
            configs="../configs/train_triplane_vae.yaml",
            output_dir=os.path.join(base_dir, "ae/chair"),
            log_dir=os.path.join(base_dir, "ae/chair"),
            num_workers=8,
            batch_size=22,
            epochs=200,
            warmup_epochs=5,
            dist_eval=True,
            category="chair",
            data_path=data_path,
            replica=5,
            clip_grad=0.35,
            port=15000,
            gpus=gpus
        )

    if train_type in ['diffusion', 'both']:
        # Train Triplane Diffusion
        run_training(
            script="train_triplane_diffusion.py",
            configs="../configs/train_triplane_diffusion.yaml",
            output_dir=os.path.join(base_dir, "dm/debug"),
            log_dir=os.path.join(base_dir, "dm/debug"),
            num_workers=8,
            batch_size=22,
            epochs=1000,
            warmup_epochs=40,
            dist_eval=True,
            category="chair",
            data_path=data_path,
            replica=5,
            ae_path=os.path.join(base_dir, "ae/chair", "best-checkpoint.pth"),
            port=15004,
            gpus=gpus
        )