import os
import sys
import yaml
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
                        
                        # Normalize
                        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
                        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
                        
                        feat = {'mfccs': mfcc, 'mel_spec': mel_db}
                        
                        # Add to current chunk
                        all_features_chunks[-1].append(feat)
                        labels_cat_chunks[-1].append(row['category'])
                        labels_subcat_chunks[-1].append(row['subcategory'])
                        
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
            
            # Log to MLflow
            mlflow.log_param("n_mfcc", self.features_config['n_mfcc'])
            mlflow.log_param("n_mels", self.features_config['n_mels'])
            mlflow.log_param("feature_chunks", len(all_features_chunks))
            
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
                
            # Log artifacts to MLflow
            mlflow.log_artifacts(artifacts_dir, artifact_path="features")
            logging.info("Feature chunks saved successfully")
            
        except Exception as e:
            logging.error(f"Failed to save feature chunks: {e}")
            raise MyException(e, sys) from e

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    with mlflow.start_run(run_name="feature_extraction"):
        # Load clips metadata
        df_clips = pd.read_csv("artifacts/data/clips_5sec/clips_metadata.csv")
        
        feature_extractor = FeatureExtraction(args.config_path)
        feature_extractor.extract_all_features(df_clips)
