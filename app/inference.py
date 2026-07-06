"""
inference.py
------------
Módulo de inferencia YOLO para detección de daños en vehículos.

Funciones principales:
    - cargar_modelo()       → carga best.pt una sola vez
    - detectar_danos()      → recibe ruta de imagen, devuelve imagen anotada + detecciones
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv
from utils import obtener_zona, nombre_clase_legible

load_dotenv()

# ============================================================
# Rutas
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "models" / "best.pt"

# Umbral de confianza por defecto (se puede sobreescribir desde .env)
CONF_DEFAULT = float(os.getenv("CONF_THRESHOLD", 0.4))

# Variable global para no recargar el modelo en cada inferencia
_modelo = None


def cargar_modelo() -> YOLO:
    """
    Carga el modelo YOLOv8 desde models/best.pt.
    Se carga una sola vez y se reutiliza (singleton).
    """
    global _modelo
    if _modelo is None:
        if not MODELO_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {MODELO_PATH}.\n"
                "Por favor entrena el modelo en Google Colab y copia best.pt a models/"
            )
        _modelo = YOLO(str(MODELO_PATH))
    return _modelo


def detectar_danos(ruta_imagen: str, conf_umbral: float = CONF_DEFAULT) -> tuple:
    """
    Ejecuta inferencia YOLO sobre una imagen y retorna los resultados.

    Args:
        ruta_imagen (str): Ruta al archivo de imagen (.jpg, .png, etc.)
        conf_umbral (float): Umbral mínimo de confianza (0.0 - 1.0)

    Returns:
        tuple: (imagen_anotada_rgb, lista_detecciones)
            - imagen_anotada_rgb: np.ndarray en formato RGB con bounding boxes dibujadas
            - lista_detecciones: lista de dicts con keys:
                {
                    "clase": str,
                    "clase_legible": str,
                    "confianza": float,
                    "zona": str,
                    "coordenadas": {"x1", "y1", "x2", "y2"}
                }

    Raises:
        FileNotFoundError: Si el modelo best.pt no existe.
        ValueError: Si la imagen no puede ser cargada.
    """
    modelo = cargar_modelo()

    # Validar que la imagen exista y sea legible
    img_path = Path(ruta_imagen)
    if not img_path.exists():
        raise ValueError(f"Imagen no encontrada: {ruta_imagen}")

    img_cv = cv2.imread(str(img_path))
    if img_cv is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")

    alto_img, ancho_img = img_cv.shape[:2]

    # Ejecutar inferencia
    results = modelo.predict(
        source=str(ruta_imagen),
        conf=conf_umbral,
        verbose=False,
    )

    result = results[0]

    # Imagen anotada con bounding boxes (YOLOv8 la genera en BGR)
    imagen_anotada_bgr = result.plot()
    imagen_anotada_rgb = cv2.cvtColor(imagen_anotada_bgr, cv2.COLOR_BGR2RGB)

    # Extraer detecciones estructuradas
    detecciones = []
    for box in result.boxes:
        clase_id = int(box.cls[0])
        clase = modelo.names[clase_id]
        confianza = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        zona = obtener_zona(x1, y1, x2, y2, ancho_img, alto_img)

        detecciones.append({
            "clase": clase,
            "clase_legible": nombre_clase_legible(clase),
            "confianza": round(confianza, 4),
            "zona": zona,
            "coordenadas": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            },
        })

    # Ordenar por confianza descendente
    detecciones.sort(key=lambda d: d["confianza"], reverse=True)

    return imagen_anotada_rgb, detecciones


def imagen_original_rgb(ruta_imagen: str) -> np.ndarray:
    """Lee y devuelve la imagen original en formato RGB para mostrar en Streamlit."""
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
