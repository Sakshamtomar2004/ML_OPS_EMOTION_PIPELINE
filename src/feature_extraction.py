import os
import sys
import yaml
import json
import pickle
import numpy as np
import librosa
import mlflow
from tqdm import tqdm
from src.logger import logging
from src.exception import MyException

class FeatureExtraction:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.features_config = self.config['features']
        self.audio_config = self.config['audio']

    def extract_all_features(self, df):
        """Extract MFCC and Mel-spectrogram features with chunking"""
        try:
            logging.info("Starting feature extraction")
            
            MAX_BYTES = self.features_config['max_feature_bytes']
            all_features_chunks = [[]]
            labels_cat_chunks = [[]]
            labels_subcat_chunks = [[]]
            current_bytes = 0
            total_features = 0

            for subcat, group in df.groupby('subcategory'):
                for idx, row in tqdm(group.iterrows(),
                                   total=len(group),
                                   desc=f"Extracting {subcat}"):
                    
                    try:
                        y, sr = librosa.load(
                            row['file_path'],
                            sr=self.audio_config['sample_rate'],
                            mono=True
                        )
                        
                        # Extract features
                        mfcc = librosa.feature.mfcc(
                            y=y, sr=sr,
                            n_mfcc=self.features_config['n_mfcc'],
                            n_fft=self.features_config['n_fft'],
                            hop_length=self.features_config['hop_length']
                        )
                        
                        mel = librosa.feature.melspectrogram(
                            y=y, sr=sr,
                            n_mels=self.features_config['n_mels'],
                            n_fft=self.features_config['n_fft'],
                            hop_length=self.features_config['hop_length']
                        )
                        mel_db = librosa.power_to_db(mel, ref=np.max)
                        
                        # Standardization
                        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
                        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
                        
                        feat = {'mfccs': mfcc, 'mel_spec': mel_db}
                        
                        # Add to current chunk
                        all_features_chunks[-1].append(feat)
                        labels_cat_chunks[-1].append(row['category'])
                        labels_subcat_chunks[-1].append(row['subcategory'])
                        total_features += 1
                        
                        # Check chunk size
                        current_bytes += feat['mfccs'].nbytes + feat['mel_spec'].nbytes
                        if current_bytes >= MAX_BYTES:
                            all_features_chunks.append([])
                            labels_cat_chunks.append([])
                            labels_subcat_chunks.append([])
                            current_bytes = 0
                            
                    except Exception as e:
                        logging.error(f"Failed to extract features from {row['file_path']}: {e}")
                        continue

            # Save feature chunks
            self.save_feature_chunks(all_features_chunks, labels_cat_chunks, labels_subcat_chunks)
            
            # Save metrics for DVC
            metrics = {
                "total_features": total_features,
                "feature_chunks": len(all_features_chunks),
                "n_mfcc": self.features_config['n_mfcc'],
                "n_mels": self.features_config['n_mels'],
                "chunk_size_mb": MAX_BYTES / (1024**2)
            }
            os.makedirs("artifacts/metrics", exist_ok=True)
            with open("artifacts/metrics/feature_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Log to MLflow if in active run
            if mlflow.active_run():
                mlflow.log_param("n_mfcc", self.features_config['n_mfcc'])
                mlflow.log_param("n_mels", self.features_config['n_mels'])
                mlflow.log_param("feature_chunks", len(all_features_chunks))
                mlflow.log_metric("total_features", total_features)
            
            logging.info(f"Feature extraction completed. Created {len(all_features_chunks)} chunks")
            return all_features_chunks, labels_cat_chunks, labels_subcat_chunks
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            raise MyException(e, sys) from e

    def save_feature_chunks(self, features_chunks, cats_chunks, subcats_chunks):
        """Save feature chunks to artifacts"""
        try:
            artifacts_dir = "artifacts/features"
            os.makedirs(artifacts_dir, exist_ok=True)
            
            with open(os.path.join(artifacts_dir, "features_chunks.pkl"), "wb") as f:
                pickle.dump(features_chunks, f)
            with open(os.path.join(artifacts_dir, "cats_chunks.pkl"), "wb") as f:
                pickle.dump(cats_chunks, f)
            with open(os.path.join(artifacts_dir, "subcats_chunks.pkl"), "wb") as f:
                pickle.dump(subcats_chunks, f)
                
            # Log artifacts to MLflow if in active run
            if mlflow.active_run():
                mlflow.log_artifacts(artifacts_dir, artifact_path="features")
            
            logging.info("Feature chunks saved successfully")
            
        except Exception as e:
            logging.error(f"Failed to save feature chunks: {e}")
            raise MyException(e, sys) from e

    def load_feature_chunks(self, features_dir="artifacts/features"):
        """Load feature chunks from pickle files"""
        try:
            with open(os.path.join(features_dir, "features_chunks.pkl"), "rb") as f:
                features_chunks = pickle.load(f)
            with open(os.path.join(features_dir, "cats_chunks.pkl"), "rb") as f:
                cats_chunks = pickle.load(f)
            with open(os.path.join(features_dir, "subcats_chunks.pkl"), "rb") as f:
                subcats_chunks = pickle.load(f)
            
            logging.info(f"Successfully loaded {len(features_chunks)} feature chunks")
            return features_chunks, cats_chunks, subcats_chunks
            
        except Exception as e:
            logging.error(f"Failed to load feature chunks: {e}")
            raise MyException(e, sys) from e

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    # Load clips metadata
    df_clips = pd.read_csv("artifacts/data/clips_5sec/clips_metadata.csv")
    
    feature_extractor = FeatureExtraction(args.config_path)
    feature_extractor.extract_all_features(df_clips)
