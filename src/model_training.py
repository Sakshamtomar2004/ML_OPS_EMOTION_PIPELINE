import os
import sys
import yaml
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.layers import Input, Conv2D, Resizing, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from src.logger import logging
from src.exception import MyException

class ModelTraining:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config['model']

    def load_features(self):
        """Load feature chunks"""
        try:
            with open("artifacts/features/features_chunks.pkl", "rb") as f:
                features_chunks = pickle.load(f)
            with open("artifacts/features/cats_chunks.pkl", "rb") as f:
                cats_chunks = pickle.load(f)
            with open("artifacts/features/subcats_chunks.pkl", "rb") as f:
                subcats_chunks = pickle.load(f)
            
            # Use first chunk for simplicity (you can extend this)
            features = features_chunks[0]#Adding [0] here
            labels_category = cats_chunks[0]
            labels_subcategory = subcats_chunks[0]
            
            return features, labels_category, labels_subcategory
            
        except Exception as e:
            logging.error(f"Failed to load features: {e}")
            raise MyException(e, sys) from e

    def prepare_data(self, features, labels_category, labels_subcategory):
        """Prepare training data"""
        try:
            X_mfcc = np.array([f['mfccs'] for f in features])
            
            category_set = sorted(set(labels_category))
            subcategory_set = sorted(set(labels_subcategory))
            
            category_to_idx = {cat: idx for idx, cat in enumerate(category_set)}
            subcategory_to_idx = {subcat: idx for idx, subcat in enumerate(subcategory_set)}
            
            y_category = np.array([category_to_idx[cat] for cat in labels_category])
            y_subcategory = np.array([subcategory_to_idx[subcat] for subcat in labels_subcategory])
            
            y_category_onehot = tf.keras.utils.to_categorical(y_category, num_classes=len(category_set))
            y_subcategory_onehot = tf.keras.utils.to_categorical(y_subcategory, num_classes=len(subcategory_set))
            
            # Train-test split
            X_train, X_test, y_train_cat, y_test_cat, y_train_subcat, y_test_subcat = train_test_split(
                X_mfcc, y_category_onehot, y_subcategory_onehot,
                test_size=self.model_config['validation_split'],
                random_state=self.config['random_state'],
                stratify=y_category
            )
            
            # Add channel dimension
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train_cat': y_train_cat, 'y_test_cat': y_test_cat,
                'y_train_subcat': y_train_subcat, 'y_test_subcat': y_test_subcat,
                'category_mapping': {idx: cat for cat, idx in category_to_idx.items()},
                'subcategory_mapping': {idx: subcat for subcat, idx in subcategory_to_idx.items()}
            }
            
        except Exception as e:
            logging.error(f"Data preparation failed: {e}")
            raise MyException(e, sys) from e

    def build_model(self, input_shape, num_categories, num_subcategories):
        """Build ConvNeXt-based model"""
        try:
            base_model = ConvNeXtTiny(
                include_top=False, input_shape=(224, 224, 3), weights="imagenet"
            )
            base_model.trainable = False

            input_layer = Input(shape=input_shape)
            x = Resizing(224, 224)(input_layer)
            x = Conv2D(3, (3,3), padding="same", activation="relu")(x)
            x = base_model(x, training=False)
            x = GlobalAveragePooling2D()(x)

            shared_features = Dense(256, activation="relu")(x)

            category_output = Dense(128, activation="relu")(shared_features)
            category_output = Dropout(self.model_config['dropout_rate'])(category_output)
            category_output = Dense(num_categories, activation="softmax", name="category_output")(category_output)

            subcategory_output = Dense(128, activation="relu")(shared_features)
            subcategory_output = Dropout(self.model_config['dropout_rate'])(subcategory_output)
            subcategory_output = Dense(num_subcategories, activation="softmax", name="subcategory_output")(subcategory_output)

            model = Model(inputs=input_layer, outputs=[category_output, subcategory_output])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_config['learning_rate']),
                loss={
                    'category_output': 'categorical_crossentropy',
                    'subcategory_output': 'categorical_crossentropy'
                },
                metrics={
                    'category_output': 'accuracy',
                    'subcategory_output': 'accuracy'
                }
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Model building failed: {e}")
            raise MyException(e, sys) from e

    def train_model(self, model, prepared_data):
        """Train the model and log to MLflow"""
        try:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=self.model_config['patience'],
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.model_config['reduce_lr_patience'],
                    min_lr=self.model_config['min_lr']
                )
            ]

            history = model.fit(
                prepared_data['X_train'],
                {
                    'category_output': prepared_data['y_train_cat'],
                    'subcategory_output': prepared_data['y_train_subcat']
                },
                validation_data=(
                    prepared_data['X_test'],
                    {
                        'category_output': prepared_data['y_test_cat'],
                        'subcategory_output': prepared_data['y_test_subcat']
                    }
                ),
                epochs=self.model_config['epochs'],
                batch_size=self.model_config['batch_size'],
                callbacks=callbacks
            )

            # Plot and save training curves
            self.plot_training_curves(history)
            
            # Log metrics to MLflow
            final_loss = min(history.history['val_loss'])
            final_cat_acc = max(history.history['val_category_output_accuracy'])
            final_subcat_acc = max(history.history['val_subcategory_output_accuracy'])
            
            mlflow.log_metric("val_loss", final_loss)
            mlflow.log_metric("val_category_accuracy", final_cat_acc)
            mlflow.log_metric("val_subcategory_accuracy", final_subcat_acc)
            
            return model, history
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise MyException(e, sys) from e

    def plot_training_curves(self, history):
        """Plot and save training curves"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Category loss
            axes[0,0].plot(history.history['category_output_loss'], label='Train Loss')
            axes[0,0].plot(history.history['val_category_output_loss'], label='Val Loss')
            axes[0,0].set_title('Category Loss')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Category accuracy
            axes[0,1].plot(history.history['category_output_accuracy'], label='Train Acc')
            axes[0,1].plot(history.history['val_category_output_accuracy'], label='Val Acc')
            axes[0,1].set_title('Category Accuracy')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Subcategory loss
            axes[1,0].plot(history.history['subcategory_output_loss'], label='Train Loss')
            axes[1,0].plot(history.history['val_subcategory_output_loss'], label='Val Loss')
            axes[1,0].set_title('Subcategory Loss')
            axes[1,0].legend()
            axes[1,0].grid(True)
            
            # Subcategory accuracy
            axes[1,1].plot(history.history['subcategory_output_accuracy'], label='Train Acc')
            axes[1,1].plot(history.history['val_subcategory_output_accuracy'], label='Val Acc')
            axes[1,1].set_title('Subcategory Accuracy')
            axes[1,1].legend()
            axes[1,1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs("artifacts/plots", exist_ok=True)
            plot_path = "artifacts/plots/training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Log to MLflow
            mlflow.log_artifact(plot_path)
            logging.info("Training curves saved and logged to MLflow")
            
        except Exception as e:
            logging.error(f"Failed to plot training curves: {e}")
            raise MyException(e, sys) from e

    def save_model(self, model, prepared_data):
        """Save model and metadata"""
        try:
            os.makedirs("artifacts/models", exist_ok=True)
            
            # Save model
            model_path = "artifacts/models/emotion_classifier.h5"
            model.save(model_path)
            
            # Save metadata
            metadata = {
                'category_mapping': prepared_data['category_mapping'],
                'subcategory_mapping': prepared_data['subcategory_mapping'],
                'input_shape': prepared_data['X_train'].shape[1:],
                'config': self.config
            }
            
            metadata_path = "artifacts/models/metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Log model to MLflow
            mlflow.tensorflow.log_model(model, "model")
            mlflow.log_artifact(metadata_path)
            
            logging.info("Model and metadata saved successfully")
            
        except Exception as e:
            logging.error(f"Model saving failed: {e}")
            raise MyException(e, sys) from e

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    with mlflow.start_run(run_name="model_training"):
        trainer = ModelTraining(args.config_path)
        
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
