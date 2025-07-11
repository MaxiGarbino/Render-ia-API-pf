# app/services/predictor.py
import numpy as np
from tensorflow.keras.preprocessing import image
from app.models.model_loader import models, class_labels

async def predict_image(img_path: str) -> dict:
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    all_predictions = [model.predict(x)[0] for model in models]
    averaged = np.mean(all_predictions, axis=0)

    sorted_indices = np.argsort(averaged)[::-1]
    predictions = [
        {"label": class_labels[i], "probability": float(averaged[i])}
        for i in sorted_indices
    ]
    return {"predicciones": predictions}
