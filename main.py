import os
import logging
from tqdm import tqdm
from parse_args import parse_arguments
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle
from experiments.baseline import BaselineExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment
from plot import plot_loss

def setup_experiment(opt):
    
    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_baseline(opt)
        
    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_domain_disentangle(opt)

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        if opt['clip_finetune']:
            train_loader, validation_loader, test_loader, train_clip_loader = build_splits_clip_disentangle(opt)
            return experiment, train_loader, validation_loader, test_loader, train_clip_loader
        else:
            train_loader, validation_loader, test_loader = build_splits_clip_disentangle(opt)
    else:
        raise ValueError('Experiment not yet supported.')

    return experiment, train_loader, validation_loader, test_loader

def main(opt):
    if opt["clip_finetune"]:
        experiment, train_loader, validation_loader, test_loader, train_clip_loader = setup_experiment(opt)
    else:
        experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)

    if not opt['test']: # Skip training if '--test' flag is set
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0
        total_clip_loss = 0
        iteration_log = list()
        train_log = list()
        validation_log = list()

        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)

        if opt["clip_finetune"]:   
            print("CLIP training started.")
            with tqdm(total= opt['max_iterations'] ) as pbar:

                # Train loop CLIP
                while iteration < opt['max_iterations']:

                    for data in train_clip_loader:

                        if (opt['debug']):
                            total_clip_loss += experiment.train_clip_iteration(data, debug = True, i = iteration)
                        else :
                            total_clip_loss += experiment.train_clip_iteration(data)

                        if iteration % opt['print_every'] == 0:
                            logging.info(f'[TRAIN CLIP - {iteration}] Loss CLIP: {total_clip_loss / (iteration + 1)}')               

                        iteration += 1
                        if iteration > opt['max_iterations']:
                            break

                        pbar.update(1)     
            print("CLIP training finished.")
        with tqdm(total= opt['max_iterations'] ) as pbar:
            print("Model training started.")
            iteration = 0
            # Train loop
            while iteration < opt['max_iterations']:
                for data in train_loader:

                    if (opt['debug']):
                        total_train_loss += experiment.train_iteration(data, debug = True, i = iteration)
                    else :
                        total_train_loss += experiment.train_iteration(data)

                    if iteration % opt['print_every'] == 0:
                        logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                
                    if iteration % opt['validate_every'] == 0:
                        # Run validation
                        train_loss = experiment.train_iteration(data)

                        if (opt['debug']):
                            val_accuracy, val_loss = experiment.validate(validation_loader, debug = True, i = iteration)
                        else :
                            val_accuracy, val_loss = experiment.validate(validation_loader)
                        logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                        
                        iteration_log.append(iteration)
                        train_log.append(train_loss)
                        validation_log.append(val_loss)                        
                        
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                        experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                    iteration += 1
                    if iteration > opt['max_iterations']:
                        break

                    pbar.update(1)
        
        if opt["plot"]:
            plot_loss(train_log, validation_log, iteration_log)

    # Test
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    test_accuracy, _ = experiment.validate(test_loader)
    logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
    print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')

if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)
