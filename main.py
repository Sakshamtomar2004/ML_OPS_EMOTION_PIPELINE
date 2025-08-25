import os
import sys
import json
import mlflow
import argparse
import yaml
from src.logger import logging
from src.exception import MyException
from src.data_ingestion import DataIngestion
from src.feature_extraction import FeatureExtraction
from src.model_training import ModelTraining
import pandas as pd

def run_pipeline(config_path):
    """Run the complete DVC+MLflow pipeline"""
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set MLflow experiment
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
        with mlflow.start_run(run_name=config['mlflow']['run_name']):
            logging.info("Starting emotion detection pipeline")
            
            # Log all parameters to MLflow
            mlflow.log_params(config['audio'])
            mlflow.log_params(config['features'])
            mlflow.log_params(config['model'])
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion(config_path)
            data_ingestion.download_data()
            df_clips = data_ingestion.clip_5sec_segments()
            
            # Step 2: Feature Extraction
            logging.info("Step 2: Feature Extraction")
            feature_extractor = FeatureExtraction(config_path)
            features_chunks, cats_chunks, subcats_chunks = feature_extractor.extract_all_features(df_clips)
            
            # Step 3: Model Training
            logging.info("Step 3: Model Training")
            trainer = ModelTraining(config_path)
            
            # Load features
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
            
            # Save final pipeline metrics for DVC
            pipeline_metrics = {
                "final_val_loss": min(history.history.get('val_loss', [0])),
                "final_category_acc": max(history.history.get('val_category_output_accuracy', )),
                "final_subcategory_acc": max(history.history.get('val_subcategory_output_accuracy', )),
                "total_epochs": len(history.history.get('loss', [])),
                "model_params": model.count_params()
            }
            
            os.makedirs("artifacts/metrics", exist_ok=True)
            with open("artifacts/metrics/pipeline_metrics.json", "w") as f:
                json.dump(pipeline_metrics, f, indent=2)
            
            logging.info("Pipeline completed successfully!")
            
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise MyException(e, sys) from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    run_pipeline(args.config_path)
