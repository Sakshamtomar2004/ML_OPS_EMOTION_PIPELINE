import os
import sys
import yaml
import mlflow
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from src.logger import logging
from src.exception import MyException

class DataIngestion:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_config = self.config['data']
        self.audio_config = self.config['audio']

    def download_data(self):
        """Download raw audio data"""
        try:
            logging.info("Starting data download")
            import opendatasets as od
            od.download(
                self.data_config['raw_data_url'],
                data_dir=self.data_config['raw_data_dir']
            )
            logging.info("Data download completed")
            mlflow.log_param("data_source", self.data_config['raw_data_url'])
            return True
        except Exception as e:
            logging.error(f"Data download failed: {e}")
            raise MyException(e, sys) from e

    def clip_5sec_segments(self):
        """Create 5-second audio clips"""
        try:
            logging.info("Starting audio clipping to 5-second segments")

            root_dir = self.data_config['raw_data_dir']
            root_dir=os.path.join(root_dir,'emotions-audio-clips')
            output_dir = self.data_config['clips_data_dir']
            sample_rate = self.audio_config['sample_rate']
            clip_duration = self.audio_config['clip_duration']
            clip_samples = int(clip_duration * sample_rate)

            records = []

            for category in ["Danger", "Non-Danger"]:
                cat_path = os.path.join(root_dir, category)
                if not os.path.isdir(cat_path):
                    continue

                for subcat in sorted(os.listdir(cat_path)):
                    sub_dir = os.path.join(cat_path, subcat)
                    if not os.path.isdir(sub_dir):
                        continue

                    remainder = np.array([], dtype=np.float32)
                    clip_idx = 0
                    out_subdir = os.path.join(output_dir, category, subcat)
                    os.makedirs(out_subdir, exist_ok=True)

                    for fname in tqdm(sorted(os.listdir(sub_dir)),
                                      desc=f"{category}/{subcat}", unit="file"):
                        if not fname.lower().endswith(('.wav', '.mp3')):
                            continue

                        file_path = os.path.join(sub_dir, fname)
                        try:
                            y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
                        except Exception as e:
                            logging.error(f"Failed to load {file_path}: {e}")
                            continue

                        # Exact 5-second files
                        duration = len(y) / sr
                        if abs(duration - clip_duration) < 1e-2:
                            out_name = fname
                            out_path = os.path.join(out_subdir, out_name)
                            sf.write(out_path, y, samplerate=sr)
                            records.append({
                                "file_path": out_path,
                                "category": category,
                                "subcategory": subcat
                            })
                            #logging.info(f"Added exact {clip_duration}s file: {fname}")
                            continue

                        # Prepend remainder
                        if remainder.size:
                            y = np.concatenate([remainder, y])
                            remainder = np.array([], dtype=np.float32)

                        idx = 0
                        total = len(y)
                        while idx + clip_samples <= total:
                            clip = y[idx:idx + clip_samples]
                            out_name = f"{os.path.splitext(fname)[0]}_clip{clip_idx:04d}.wav"
                            out_path = os.path.join(out_subdir, out_name)
                            sf.write(out_path, clip, samplerate=sr)
                            records.append({
                                "file_path": out_path,
                                "category": category,
                                "subcategory": subcat
                            })
                            clip_idx += 1
                            idx += clip_samples

                        remainder = y[idx:]

            # If no clips were created, list all original short files
            if not records:
                data = []
                for category in os.listdir(root_dir):
                    cat_path = os.path.join(root_dir, category)
                    if not os.path.isdir(cat_path):
                        continue
                    for subcat in os.listdir(cat_path):
                        sub_dir = os.path.join(cat_path, subcat)
                        if not os.path.isdir(sub_dir):
                            continue
                        for fname in os.listdir(sub_dir):
                            if fname.lower().endswith(('.wav', '.mp3', '.ogg')):
                                data.append({
                                    "file_path": os.path.join(sub_dir, fname),
                                    "category": category,
                                    "subcategory": subcat
                                })
                df = pd.DataFrame(data, columns=["file_path", "category", "subcategory"])
            else:
                df = pd.DataFrame(records, columns=["file_path", "category", "subcategory"])

            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "clips_metadata.csv")
            df.to_csv(csv_path, index=False)

            mlflow.log_param("total_clips", len(df))
            mlflow.log_param("clip_duration", clip_duration)
            mlflow.log_artifact(csv_path)
         #   mlflow.log_artifact("src\data_ingestion.py")
         #   mlflow.log_artifact("src\feature_extraction.py")


            logging.info(f"Saved metadata CSV with {len(df)} clips: {csv_path}")
            return df

        except Exception as e:
            logging.error(f"Audio clipping failed: {e}")
            raise MyException(e, sys) from e


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/config.yaml"
    )
    args = parser.parse_args()

    with mlflow.start_run(run_name="data_ingestion"):
        ing = DataIngestion(args.config_path)
        ing.download_data()
        ing.clip_5sec_segments()
        mlflow.log_artifact("src\data_ingestion.py")
        mlflow.log_artifact("src\feature_extraction.py")

