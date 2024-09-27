import argparse
import logging
import os
from dataloader.dataloader import data_loader
from model.model import get_cct_model
from train.train import logo_train, loso_train

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def parse_args():
    """
    Parse command line arguments to determine which training type (LOSO/LOGO) to run.
    """
    parser = argparse.ArgumentParser(description='Train a model using LOGO or LOSO strategy.')
    parser.add_argument('--train_type', choices=['loso', 'logo'], required=True,
                        help='Specify whether to train using Leave-One-Subject-Out (loso) or Leave-One-Group-Out (logo).')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing participant data.')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory to save model checkpoints.')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    # Ensure the checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Load the model
    model = get_cct_model()

    # Check training type
    if args.train_type == 'loso':
        # Perform LOSO training
        logging.info("Starting LOSO training...")
        # Prepare LOSO datasets
        loso_datasets = data_loader.prepare_loso_dataset(args.base_dir)
        # Train with LOSO
        for group_index, (train_participants, val_participant, test_participant) in enumerate(loso_datasets, start=1):
            identifier = f"LOSO_Group{group_index}_{test_participant}"
            # Prepare training, validation, and test data
            train_data = [os.path.join(args.base_dir, participant, 'data') for participant in train_participants]
            val_data = [os.path.join(args.base_dir, participant, 'data') for participant in val_participant]
            test_data = os.path.join(args.base_dir, test_participant, 'data')
            # Call the LOSO training function
            loso_train.run_loso_training(model, train_data, val_data, test_data, args.checkpoint_dir, identifier)

    elif args.train_type == 'logo':
        # Perform LOGO training
        logging.info("Starting LOGO training...")
        # Define sub-group structure
        sub_group_definitions = {
            1: [['On_9', 'On_3', 'On_6'], ['On_1', 'On_2', 'On_4'], ['On_5', 'On_7', 'On_8']],
            2: [['Off_4', 'Off_1', 'Off_7'], ['Off_2', 'Off_3', 'Off_10'], ['Off_5', 'Off_6', 'Off_9']],
            3: [['On_9', 'Off_1', 'On_6', 'Off_4'], ['Off_10', 'On_2', 'On_4', 'Off_7'], ['On_7', 'On_5', 'Off_5', 'Off_3'], ['On_1', 'Off_2', 'Off_6', 'On_8']]
        }
        # Prepare LOGO datasets
        logo_datasets = data_loader.prepare_logo_dataset(args.base_dir, sub_group_definitions)
        # Train with LOGO
        for main_group_index, (train_participants, validation_participants, test_participant) in enumerate(logo_datasets, start=1):
            identifier = f"LOGO_MainGroup{main_group_index}_{'_'.join(test_participant)}"
            # Prepare training, validation, and test data
            train_data = [os.path.join(args.base_dir, participant, 'data') for participant in train_participants]
            val_data = [os.path.join(args.base_dir, participant, 'data') for participant in validation_participants]
            test_data = [os.path.join(args.base_dir, participant, 'data') for participant in test_participant]
            # Call the LOGO training function
            logo_train.run_logo_training(model, train_data, val_data, test_data, args.checkpoint_dir, identifier)
if __name__ == "__main__":
    main()
