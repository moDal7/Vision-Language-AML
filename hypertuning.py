import os
import logging
from tqdm import tqdm
from parse_args import parse_arguments
from load_data import build_splits_validation
from experiments.domain_disentangle_tuning import DomainDisentangleExperiment
from plot import plot_loss

VALUES_TO_TEST = [0.01, 0.1, 1, 10, 100]
NUM_WEIGHTS = 5
SPLITS = 2

def main(opt):

    for i in range(NUM_WEIGHTS):
        for weight in VALUES_TO_TEST:

            weights = [1, 1, 1, 1, 1]
            weights[i] = weight
            total_val_accuracy = 0
            total_val_loss = 0

            for i in range(SPLITS):
                if i == 0:
                    train_loader, validation_loader = build_splits_validation(opt)
                else:
                    validation_loader, train_loader = build_splits_validation(opt)

                experiment = DomainDisentangleExperiment(opt, weights)
                iteration = 0
                best_accuracy = 0
                total_train_loss = 0
                iteration_log = list()
                train_log = list()
                val_log = list()
                final_val_log = list()
                logging.info(opt)
                
                with tqdm(total=opt["max_iterations"]) as pbar:

                    # Train loop
                    while iteration < opt['max_iterations']:
                        for data in train_loader:

                            total_train_loss += experiment.train_iteration(data)

                            if iteration % opt['print_every'] == 0:
                                logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                        
                                if iteration % opt['validate_every'] == 0:
                                    # Run validation
                                    train_loss = experiment.train_iteration(data)
                                    iteration_log.append(iteration)
                                    train_log.append(train_loss)

                                iteration += 1
                                if iteration > opt['max_iterations']:
                                    break

                                pbar.update(1)
                
                if opt["plot"]:
                    plot_loss(train_log, iteration_log)

                val_accuracy, val_loss = experiment.validate(validation_loader)
                val_log.append(val_accuracy, val_loss, [weights])
                total_val_accuracy += val_accuracy
                total_val_loss += val_loss

            total_val_accuracy = total_val_accuracy/SPLITS
            total_val_loss = total_val_loss/SPLITS
            final_val_log.append(total_val_accuracy, total_val_loss, weights)
            logging.info(f'[WEIGHTS]: {weights}, [VAL] Accuracy: {(100 * total_val_accuracy):.2f}')
    final_val_log.sort(key=final_val_log[0], reverse=True)
    logging.info(f'Best performing weights: {final_val_log[0][2]}, ACCURACY {(100 * final_val_log[0][0]):.2f}')
    print(f'Best performing weights: {final_val_log[0][2]}, ACCURACY {(100 * final_val_log[0][0]):.2f}')

if __name__ == '__main__':

    opt = parse_arguments()
    opt['output_path_hyper'] = f'{opt["output_path"]}/record/hyperparam_tuning'
    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path_hyper"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)
