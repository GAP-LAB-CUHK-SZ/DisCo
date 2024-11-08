import os
import subprocess
import argparse

def run_command(cmd):
    subprocess.run(' '.join(cmd), shell=True)

def run_operation(script, configs, data_path, gpus, batch_size=None , output_dir=None, log_dir=None,
                  epochs=None, warmup_epochs=None, dist_eval=False, category=None, 
                    clip_grad=None, ae_path=None, dm_path=None, port=15000, finetune=None, finetune_path=None,replica=None,output_folder=None,
                  eval_cd=None,reso=None,save_mesh=None,save_par_points=None,save_image=None,save_surface=None,blr=None):
    print("num of gpus",len(gpus.split(',')))
    cmd = [
        'CUDA_VISIBLE_DEVICES=' + gpus,
        'python -m torch.distributed.run',
        '--master_port', str(port),
        '--nproc_per_node=' + str(len(gpus.split(','))),
        script,
        '--configs', configs,
        '--data-pth', data_path
    ]
    if batch_size:
        cmd.extend(["--batch_size",str(batch_size)])
    if output_dir:
        cmd.extend(['--output_dir', output_dir])
    if log_dir:
        cmd.extend(['--log_dir', log_dir])
    if epochs:
        cmd.extend(['--epochs', str(epochs)])
    if warmup_epochs:
        cmd.extend(['--warmup_epochs', str(warmup_epochs)])
    if category:
        cmd.extend(['--category', category])
    if dist_eval:
        cmd.append('--dist_eval')
    if clip_grad is not None:
        cmd.extend(['--clip_grad', str(clip_grad)])
    if ae_path:
        cmd.extend(['--ae-pth', ae_path])
    if dm_path:
        cmd.extend(['--dm-pth', dm_path])
    if output_folder:
        cmd.extend(['--output_folder',output_folder])
    if finetune:
        cmd.extend(['--finetune'])
    if finetune_path:
        cmd.extend(['--finetune-pth', finetune_path])
    if replica:
        cmd.extend(['--replica', str(replica)])
    if eval_cd:
        cmd.append('--eval_cd')
    if reso:
        cmd.extend(['--reso', str(reso)])
    if save_mesh:
        cmd.append('--save_mesh')
    if save_par_points:
        cmd.append('--save_par_points')
    if save_image:
        cmd.append('--save_image')
    if save_surface:
        cmd.append('--save_surface')
    if blr:
        cmd.extend(['--blr', str(blr)])

    print(cmd)

    run_command(cmd)

def train_vae(args, category):
    run_operation(
        script="disco/scripts/train_triplane_vae.py",
        configs="disco/configs/train_triplane_vae.yaml",
        output_dir=os.path.join(args.base_dir, f"ae/{category}"),
        log_dir=os.path.join(args.base_dir, f"ae/{category}"),
        batch_size=22,
        epochs=200,
        warmup_epochs=5,
        dist_eval=True,
        category=category,
        data_path=args.data_path,
        clip_grad=0.35,
        port=15000,
        gpus=args.gpus,
        replica=5, #can choose to replicate the dataset more, if the number of samples is small such as shelf category
    )

def cache_triplane_features(args, category):
    run_operation(
        script="disco/scripts/cache_triplane_vae_features.py",
        configs="disco/configs/train_triplane_vae.yaml",
        batch_size=10,
        data_path=args.data_path,
        gpus=args.gpus,
        category=category,
        ae_path=os.path.join(args.base_dir, f"ae/{category}/best-checkpoint.pth"),
        port=15000
    )
    
def cache_image_features(args, category):
    run_operation(
        script="disco/scripts/cache_img_vit_features.py",
        configs='disco/configs/train_triplane_diffusion.yaml',
        batch_size=24,
        data_path=args.data_path,
        gpus=args.gpus,
        category=category,
        port=15000
    )
    
def train_diffusion(args, category):
    run_operation(
        script="disco/scripts/train_triplane_diffusion.py",
        configs="disco/configs/train_triplane_diffusion.yaml",
        output_dir=os.path.join(args.base_dir, f"dm/{category}"),
        log_dir=os.path.join(args.base_dir, f"dm/{category}"),
        batch_size=22,
        epochs=1000,
        warmup_epochs=40,
        dist_eval=True,
        category=category,
        data_path=args.data_path,
        ae_path=os.path.join(args.base_dir, f"ae/{category}", "best-checkpoint.pth"),
        port=15004,
        gpus=args.gpus,
        replica = 5 #can choose to replicate the dataset more, if the number of samples is small such as shelf category
    )

def finetune_diffusion(args, category):
    run_operation(
        script="disco/scripts/train_triplane_diffusion.py",
        configs="disco/configs/finetune_triplane_diffusion.yaml",
        output_dir=os.path.join(args.base_dir, f"finetune_dm/{category}"),
        log_dir=os.path.join(args.base_dir, f"finetune_dm/{category}"),
        batch_size=22,
        epochs=500,
        warmup_epochs=20,
        blr=0.0001,
        dist_eval=True,
        category=category,
        data_path=args.data_path,
        ae_path=os.path.join(args.base_dir, f"ae/{category}", "best-checkpoint.pth"),
        port=15004,
        gpus=args.gpus,
        finetune=True,
        finetune_path=os.path.join(args.base_dir, f"dm/{category}/best-checkpoint.pth"),
        replica=5, #can choose to replicate the dataset more, if the number of samples is small such as shelf category
    )

def evaluate(args,category):
    run_operation(
        script="disco/scripts/evaluate_reconstruction.py",
        configs="disco/configs/finetune_triplane_diffusion.yaml",
        output_folder=os.path.join(f"results/{category}"),
        category=category,
        data_path=args.data_path,
        ae_path=os.path.join(args.base_dir, f"ae/{category}", "best-checkpoint.pth"),
        dm_path=os.path.join(args.base_dir, f"finetune_dm/{category}", "best-checkpoint.pth"),
        port=15004,
        gpus=args.gpus,
        eval_cd=True,
        reso=256,
        save_mesh=True,
        save_par_points=True,
        save_image=True,
        save_surface=True,
    )

def main():
    parser = argparse.ArgumentParser(description='Launch Triplane Model Training')
    parser.add_argument('--data_path', type=str, default="data", help='Path to the dataset')
    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7", help='Visible GPUs, default: "0,1,2,3,4,5,6,7"')
    parser.add_argument('--mode', type=str, choices=['train_vae', 'cache_triplane_features', 'cache_image_features', 
                                                     'train_diffusion', 'finetune_diffusion','evaluate','all'],
                        required=True, help='Mode to run: train_vae, cache_triplane_features, cache_image_features, train_diffusion, or all')
    parser.add_argument('--category', type=str, choices=['chair', 'cabinet', 'table', 'sofa', 'bed', 'shelf', 'all'], 
                        default='chair', help='Category to train on')
    parser.add_argument('--base_dir', type=str, default="output", help='Base directory for output')

    args = parser.parse_args()

    categories = [args.category] if args.category != 'all' else ['chair', 'cabinet', 'table', 'sofa', 'bed', 'shelf']

    for category in categories:
        if args.mode in ['train_vae', 'all']:
            train_vae(args, category)
        
        if args.mode in ['cache_triplane_features', 'all']:
            cache_triplane_features(args, category)
        
        if args.mode in ['cache_image_features', 'all']:
            cache_image_features(args, category)
            
        if args.mode in ['train_diffusion', 'all']:
            train_diffusion(args, category)

        if args.mode in ["finetune_diffusion","all"]:
            finetune_diffusion(args, category)

        if args.mode in ["evaluate"]:
            evaluate(args,category)

if __name__ == "__main__":
    main()