async def diagnosticar(img_path: str) -> dict:
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(150, 150))  # Redimensionar la imagen al tamaño esperado por el modelo
    x = image.img_to_array(img)  # Convertir la imagen a un array numpy
    x = np.expand_dims(x, axis=0)  # Añadir la dimensión del lote
    x = x / 255.0  # Normalizar la imagen dividiendo por 255

    # Realizar la predicción
    predictions = model.predict(x)

    # Obtener el índice de la clase con mayor probabilidad
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    # Obtener la probabilidad de la clase predicha
    predicted_probability = predictions[0][predicted_class_index] * 100  # Convertir a porcentaje

    # Definir un umbral mínimo para considerar la predicción como válida
    threshold = 60.0  # Umbral del 60%, puedes ajustarlo según sea necesario

    # Si la probabilidad es menor al umbral, devolver "indeterminado"
    if predicted_probability < threshold:
        predicted_class_label = "indeterminado"

    # Retornar el nombre de la clase predicha (o "indeterminado") y su probabilidad
    return {
        "diagnosis": predicted_class_label,
        "probability": f"{predicted_probability:.2f}%" if predicted_class_label != "indeterminado" else "N/A",
        "all_probabilities": {class_labels[i]: f"{predictions[0][i] * 100:.2f}%" for i in range(len(class_labels))}
    }
