import os
import shutil
import uuid
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
import numpy as np
from typing import List

app = FastAPI()

# Directorio temporal para guardar imágenes
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Define las rutas de los archivos
base_dir = os.getcwd()
save_dir = os.path.join(base_dir, 'models')

# Carga el modelo TensorFlow
model_path = os.path.join(save_dir, 'clasificador_4_clases.h5')
model = tf.keras.models.load_model(model_path)

# Etiquetas de las 4 clases (modificar según las clases reales del modelo)
class_labels = ['piel-normal', 'lunar', 'melanoma', 'acne']

async def diagnosticar(img_path: str) -> dict:
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(150, 150))  # Redimensionar al tamaño esperado
    x = image.img_to_array(img)  # Convertir a un array numpy
    x = np.expand_dims(x, axis=0)  # Añadir la dimensión del lote
    x = x / 255.0  # Normalizar dividiendo por 255

    # Realizar la predicción
    predictions = model.predict(x)[0]  # Obtenemos las probabilidades por clase

    # Ordenar las probabilidades y sus índices de mayor a menor
    sorted_indices = np.argsort(predictions)[::-1]  # Índices ordenados en orden descendente
    sorted_labels = [class_labels[i] for i in sorted_indices]  # Etiquetas ordenadas
    sorted_probabilities = [predictions[i] * 100 for i in sorted_indices]  # Probabilidades en porcentaje

    # Diagnóstico inicial (primera clase)
    primary_diagnosis = sorted_labels[0]
    primary_probability = sorted_probabilities[0]

    # Verificar si la probabilidad de melanoma está por debajo del umbral del 80%
    if primary_diagnosis == "melanoma" and primary_probability < 80:
        # Cambiar el diagnóstico al segundo más probable
        secondary_diagnosis = sorted_labels[1]
        secondary_probability = sorted_probabilities[1]
        return {
            "diagnosis": secondary_diagnosis,
            "probability": f"{secondary_probability:.2f}%",
            "original_diagnosis": primary_diagnosis,
            "original_probability": f"{primary_probability:.2f}%",
            "all_probabilities": {class_labels[i]: f"{predictions[i] * 100:.2f}%" for i in range(len(class_labels))}
        }

    # Si no aplica el cambio, retornar el diagnóstico principal
    return {
        "diagnosis": primary_diagnosis,
        "probability": f"{primary_probability:.2f}%",
        "all_probabilities": {class_labels[i]: f"{predictions[i] * 100:.2f}%" for i in range(len(class_labels))}
    }


@app.post("/")
async def create_upload_file(files: List[UploadFile] = File(..., description='Subir imágenes para diagnóstico')):
    diagnoses = []

    for file in files:
        # Crear un nombre único para cada archivo
        file_id = str(uuid.uuid4())
        file_path = os.path.join(TEMP_DIR, file_id + "_" + file.filename)

        # Guardar el archivo temporalmente
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Diagnosticar la imagen
        diagnosis = await diagnosticar(file_path)
        diagnoses.append(diagnosis)

        # Eliminar la imagen después de procesar (opcional para liberar espacio)
        os.remove(file_path)

    return JSONResponse(content={"diagnoses": diagnoses})

# Endpoint adicional para manejar un solo archivo y devolver el diagnóstico
@app.post("/uploadfile/")
async def create_upload_file_single(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, file_id + "_" + file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Diagnosticar la imagen
    diagnosis = await diagnosticar(file_path)

    # Eliminar la imagen después de procesar (opcional)
    os.remove(file_path)

    return JSONResponse(content={"diagnosis": diagnosis})
