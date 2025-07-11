# app/models/model_loader.py
import os
import tensorflow as tf

MODELS_DIR = os.path.join(os.getcwd(), "models")

model_paths = [
    os.path.join(MODELS_DIR, "clasificador_4_clases.h5"),
    os.path.join(MODELS_DIR, "clasificador_4_clases_2.h5"),
    os.path.join(MODELS_DIR, "clasificador_4_clases_3.h5"),
]

models = [tf.keras.models.load_model(path) for path in model_paths]

class_labels = ['piel-normal', 'lunar', 'melanoma', 'acne']
