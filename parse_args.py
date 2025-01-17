import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument('--experiment', type=str, default='baseline', choices=['baseline', 'domain_disentangle', 'clip_disentangle'])
    parser.add_argument('--target_domain', type=str, default='cartoon', choices=['art_painting', 'cartoon', 'sketch', 'photo'])
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--max_iterations', type=int, default=5000, help='Number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--validate_every', type=int, default=100)

    # Manual weight setting for tuning
    parser.add_argument('--weights', help='List of floating point weights for the experiment.', nargs = 5, type = float)
    parser.add_argument('--weights_clip', help='List of floating point weights for the clip disentangle experiment.', nargs = 6, type = float)

    # Output settings
    parser.add_argument('--output_path', type=str, default='.', help='Where to create the output directory containing logs and weights.')
    parser.add_argument('--output_path_hyper', type=str, default='.', help='Where to create the output directory containing logs and weights.')
    parser.add_argument('--data_path', type=str, default='data/PACS', help='Locate the PACS dataset on disk.')

    # Other settings
    parser.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU.')
    parser.add_argument('--test', action='store_true', help='If set, the experiment will skip training.')
    parser.add_argument('--determ', action='store_true', help='If set, the experiment will run in deterministic mode.')
    parser.add_argument('--clip_finetune', action='store_true', help='If set, the experiment will train also the CLIP model.')
    parser.add_argument('--dg', action='store_true', help='If set, the experiment will work following the domain generalization settings.')
    parser.add_argument('--pda', action='store_true', help='If set, the experiment will work following the predictive domain adaptation settings. Needs to be used with --dg.')
    
    # Build options dict
    opt = vars(parser.parse_args())

    if not opt['cpu']:
        assert torch.cuda.is_available(), 'You need a CUDA capable device in order to run this experiment. See `--cpu` flag.'

    opt['output_path'] = f'{opt["output_path"]}/record/{opt["experiment"]}_{opt["target_domain"]}'
    opt['output_path_hyper'] = f'{opt["output_path"]}'
    
    return opt