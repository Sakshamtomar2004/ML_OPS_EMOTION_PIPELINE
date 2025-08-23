import mlflow
import argparse
import yaml
import hashlib
import json
import pandas as pd
import sys
import os

from src.logger import logging
from src.exception import MyException
from src.data_ingestion import DataIngestion
from src.feature_extraction import FeatureExtraction
from src.model_training import ModelTraining


# ---------------- Helper Functions ---------------- #

def get_config_hash(config_section: dict) -> str:
    """
    Create a hash for a given section of config.
    Ensures we can detect if parameters have changed.
    """
    config_str = json.dumps(config_section, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def save_hash(hash_val: str, hash_file: str):
    """Save hash string into a file."""
    with open(hash_file, "w") as f:
        f.write(hash_val)


def load_hash(hash_file: str) -> str:
    """Load hash string from a file."""
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return f.read().strip()
    return None


# ---------------- Pipeline ---------------- #

def run_pipeline(config_path, force=False):
    """Run the complete MLflow pipeline"""
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set MLflow experiment
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
        with mlflow.start_run(run_name=config['mlflow']['run_name']):
            logging.info("Starting emotion detection pipeline")

            # ---------------- STEP 1: DATA INGESTION ---------------- #
            clips_csv = os.path.join(config['data']['clips_data_dir'], "clips_metadata.csv")

            if os.path.exists(clips_csv) and not force:
                logging.info(f"Step 1 Skipped: Clips metadata found at {clips_csv}")
                df_clips = pd.read_csv(clips_csv)
            else:
                logging.info("Step 1: Data Ingestion started...")
                data_ingestion = DataIngestion(config_path)
                data_ingestion.download_data()
                df_clips = data_ingestion.clip_5sec_segments()
                logging.info("Step 1: Data Ingestion completed")

            # ---------------- STEP 2: FEATURE EXTRACTION ---------------- #
            feat_dir = "artifacts/features"
            os.makedirs(feat_dir, exist_ok=True)

            chunks_pkl = os.path.join(feat_dir, "features_chunks.pkl")
            hash_file = os.path.join(feat_dir, "features_config_hash.txt")

            # Compute current config hash for feature extraction
            current_hash = get_config_hash(config['features'])
            saved_hash = load_hash(hash_file)

            trainer = ModelTraining(config_path)  # Needed for feature loading

            if os.path.exists(chunks_pkl) and saved_hash == current_hash and not force:
                logging.info("Step 2 Skipped: Features already extracted and config unchanged.")
                features, labels_cat, labels_subcat = trainer.load_features()
            else:
                logging.info("Step 2: Feature Extraction started...")
                feature_extractor = FeatureExtraction(config_path)
                features_chunks, cats_chunks, subcats_chunks = feature_extractor.extract_all_features(df_clips)
                # Save hash of config
                save_hash(current_hash, hash_file)
                logging.info("Step 2: Feature Extraction completed")

            # ---------------- STEP 3: MODEL TRAINING ---------------- #
            logging.info("Step 3: Model Training started...")

            # Load features again (ensures we have them after extraction)
            features, labels_cat, labels_subcat = trainer.load_features()

            # Prepare data
            prepared_data = trainer.prepare_data(features, labels_cat, labels_subcat)

            # Build model
            input_shape = prepared_data['X_train'].shape[1:]
            num_categories = prepared_data['y_train_cat'].shape[1]
            num_subcategories = prepared_data['y_train_subcat'].shape[1]

            model = trainer.build_model(input_shape, num_categories, num_subcategories)

            # Train model
            model, history = trainer.train_model(model, prepared_data)

            # Save model
            trainer.save_model(model, prepared_data)

            logging.info("Step 3: Model Training completed")
            logging.info("Pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise MyException(e, sys) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if artifacts exist")
    args = parser.parse_args()
    
    run_pipeline(args.config_path, args.force)
