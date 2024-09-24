import os
import random
import shutil
import uuid
import logging
from sklearn.model_selection import LeaveOneOut
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LogoData:
    """Handles the preparation of data using Leave-One-Group-Out (LOGO) cross-validation."""
    @staticmethod
    def prepare_logo_dataset(base_dir: str, sub_group_definitions: dict) -> List[Tuple[List[str], List[str], List[str]]]:
        """
        Prepares datasets using Leave-One-Group-Out (LOGO) strategy.
        Args:
            base_dir (str): Base directory containing participant data.
            sub_group_definitions (dict): A dictionary defining participant sub-groups.
        Returns:
            List[Tuple[List[str], List[str], List[str]]]: A list of tuples containing train, validation, and test participants.
        """
        participants = [p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))]
        datasets = []
        for sub_groups in sub_group_definitions.values():
            for test_sub_group in sub_groups:
                remaining_sub_groups = [g for g in sub_groups if g != test_sub_group]
                for val_sub_group in remaining_sub_groups:
                    validation_participants = val_sub_group
                    train_participants = [
                        p for sub_group in remaining_sub_groups if sub_group != val_sub_group
                        for p in sub_group
                    ]
                    datasets.append((train_participants, validation_participants, test_sub_group))
        return datasets

    @staticmethod
    def copy_participant_data(participant_dirs: List[str], target_dir: str) -> None:
        """
        Copies participant data to a target directory with unique filenames.
        Args:
            participant_dirs (List[str]): List of participant directories.
            target_dir (str): Directory where files will be copied.
        """
        total_images_copied = 0
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for participant_dir in participant_dirs:
            for class_name in ['Cool', 'Neutral', 'Warm']:
                source_dir = os.path.join(participant_dir, class_name)
                target_class_dir = os.path.join(target_dir, class_name)
                if not os.path.exists(target_class_dir):
                    os.makedirs(target_class_dir)
                image_count = 0
                for filename in os.listdir(source_dir):
                    src_file = os.path.join(source_dir, filename)
                    unique_filename = f"{uuid.uuid4().hex}_{filename}"
                    dst_file = os.path.join(target_class_dir, unique_filename)
                    shutil.copy(src_file, dst_file)
                    image_count += 1
                    total_images_copied += 1
                logging.info(f"Copied {image_count} images from {source_dir} to {target_class_dir}")
        logging.info(f"Total images copied to {target_dir}: {total_images_copied}")

    @staticmethod
    def create_temp_directories(train_dirs: List[str], val_dirs: List[str], test_dirs: List[str]) -> Tuple[str, str, str]:
        """
        Creates temporary directories for training, validation, and testing datasets.
        Args:
            train_dirs (List[str]): List of training directories.
            val_dirs (List[str]): List of validation directories.
            test_dirs (List[str]): List of test directories.
        Returns:
            Tuple[str, str, str]: Paths to the temporary train, validation, and test directories.
        """
        temp_dir = 'temp_data_logo'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        train_temp_dir = os.path.join(temp_dir, 'train')
        val_temp_dir = os.path.join(temp_dir, 'val')
        test_temp_dir = os.path.join(temp_dir, 'test')
        LogoData.copy_participant_data(train_dirs, train_temp_dir)
        LogoData.copy_participant_data(val_dirs, val_temp_dir)
        LogoData.copy_participant_data(test_dirs, test_temp_dir)
        return train_temp_dir, val_temp_dir, test_temp_dir

class LosoData:
    """Handles the preparation of data using Leave-One-Subject-Out (LOSO) cross-validation."""
    @staticmethod
    def prepare_loso_dataset(base_dir: str) -> List[Tuple[List[str], str, str]]:
        """
        Prepares datasets using Leave-One-Subject-Out (LOSO) strategy.
        Args:
            base_dir (str): Base directory containing participant data.
        Returns:
            List[Tuple[List[str], str, str]]: A list of tuples containing train, validation, and test participants.
        """
        participants = [p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))]
        group_1 = [p for p in participants if p.startswith('Off')]  # Participants with mask 'Off'
        group_2 = [p for p in participants if p.startswith('On')]   # Participants with mask 'On'
        group_3 = participants  # All participants
        groups = [group_1, group_2, group_3]
        datasets = []
        for group in groups:
            loo = LeaveOneOut()
            for train_index, test_index in loo.split(group):
                train_participants = [group[i] for i in train_index]
                test_participant = group[test_index[0]]
                val_participant = train_participants[-1] if train_participants else None
                train_participants = train_participants[:-1]
                datasets.append((train_participants, val_participant, test_participant))
        return datasets

    @staticmethod
    def copy_participant_data(participant_dirs: List[str], target_dir: str) -> None:
        """
        Copies participant data to a target directory with unique filenames.
        Args:
            participant_dirs (List[str]): List of participant directories.
            target_dir (str): Directory where files will be copied.
        """
        total_images_copied = 0
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for participant_dir in participant_dirs:
            for class_name in ['Cool', 'Neutral', 'Warm']:
                source_dir = os.path.join(participant_dir, class_name)
                target_class_dir = os.path.join(target_dir, class_name)
                if not os.path.exists(target_class_dir):
                    os.makedirs(target_class_dir)
                image_count = 0
                for filename in os.listdir(source_dir):
                    src_file = os.path.join(source_dir, filename)
                    unique_filename = f"{uuid.uuid4().hex}_{filename}"
                    dst_file = os.path.join(target_class_dir, unique_filename)
                    shutil.copy(src_file, dst_file)
                    image_count += 1
                    total_images_copied += 1
                logging.info(f"Copied {image_count} images from {source_dir} to {target_class_dir}")
        logging.info(f"Total images copied to {target_dir}: {total_images_copied}")

    @staticmethod
    def create_temp_directories(train_dirs: List[str], val_dirs: List[str], test_dir: str) -> Tuple[str, str, str]:
        """
        Creates temporary directories for training, validation, and testing datasets.
        Args:
            train_dirs (List[str]): List of training directories.
            val_dirs (List[str]): List of validation directories.
            test_dir (str): Test directory.
        Returns:
            Tuple[str, str, str]: Paths to the temporary train, validation, and test directories.
        """
        temp_dir = 'temp_data_loso'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        train_temp_dir = os.path.join(temp_dir, 'train')
        val_temp_dir = os.path.join(temp_dir, 'val')
        test_temp_dir = os.path.join(temp_dir, 'test')
        LosoData.copy_participant_data(train_dirs, train_temp_dir)
        LosoData.copy_participant_data(val_dirs, val_temp_dir)
        LosoData.copy_participant_data([test_dir], test_temp_dir)
        return train_temp_dir, val_temp_dir, test_temp_dir