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
model_path = os.path.join(save_dir, 'melanoma-vs-bening.h5')
model = tf.keras.models.load_model(model_path)

async def diagnosticar(img_path: str) -> dict:
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(150, 150))  # Redimensionar la imagen al tamaño esperado por el modelo
    x = image.img_to_array(img)  # Convertir la imagen a un array numpy
    x = np.expand_dims(x, axis=0)  # Añadir la dimensión del lote
    x = x / 255.0  # Normalizar la imagen dividiendo por 255

    # Realizar la predicción
    predictions = model.predict(x)

    # Obtener la probabilidad de la clase "Melanoma"
    probability_melanoma = predictions[0][0] * 100  # Convertir a porcentaje

    # Definir un umbral para considerar la predicción válida
    threshold = 60.0  # Umbral del 60%, ajustable según sea necesario

    # Si la probabilidad de "Melanoma" es menor al umbral, devolver "indeterminado"
    if probability_melanoma < threshold:
        diagnosis = "Indeterminado"
    else:
        diagnosis = "Melanoma" if probability_melanoma >= 50 else "No hay enfermedad detectada"

    return {
        "diagnosis": diagnosis,
        "probability": f"{probability_melanoma:.2f}%"
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

        # Eliminar la imagen después de procesar
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

    # Eliminar la imagen después de procesar
    os.remove(file_path)

    return JSONResponse(content={"diagnosis": diagnosis})
