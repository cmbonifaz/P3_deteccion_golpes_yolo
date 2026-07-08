"""
inference.py
------------
Módulo de inferencia YOLO para detección de daños en vehículos.

Funciones principales:
    - cargar_modelo()         → carga best.pt una sola vez (singleton)
    - cargar_modelo_general() → carga yolov8n.pt (COCO) para validar vehículos
    - validar_vehiculo()      → verifica que la imagen contenga un vehículo
    - detectar_danos()        → recibe ruta de imagen, devuelve imagen anotada + detecciones
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
# Excepción personalizada
# ============================================================
class VehicleNotFoundError(Exception):
    """Se lanza cuando la imagen no contiene un vehículo detectable."""
    pass

# ============================================================
# Rutas
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "models" / "best.pt"

# Umbral de confianza por defecto (se puede sobreescribir desde .env)
CONF_DEFAULT = float(os.getenv("CONF_THRESHOLD", 0.4))

# Variable global para no recargar el modelo en cada inferencia
_modelo = None
_modelo_general = None  # YOLOv8n entrenado en COCO (validación de vehículos)

# Clases de vehículos en el dataset COCO
# car=2, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck
VEHICLE_CONF_MIN  = 0.15          # umbral bajo para detectar vehículos parcialmente visibles


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


def cargar_modelo_general() -> YOLO:
    """
    Carga YOLOv8n pre-entrenado en COCO para detección general.
    Se descarga automáticamente la primera vez (~6 MB).
    Se reutiliza como singleton.
    """
    global _modelo_general
    if _modelo_general is None:
        _modelo_general = YOLO("yolov8n.pt")  # descarga automática si no existe
    return _modelo_general


def validar_vehiculo(ruta_imagen: str) -> bool:
    """
    Verifica si la imagen contiene algún vehículo usando YOLOv8n (COCO).

    Args:
        ruta_imagen (str): Ruta al archivo de imagen.

    Returns:
        bool: True si se detectó al menos un vehículo, False en caso contrario.
    """
    modelo_gral = cargar_modelo_general()
    results = modelo_gral.predict(
        source=str(ruta_imagen),
        conf=VEHICLE_CONF_MIN,
        verbose=False,
    )
    for box in results[0].boxes:
        if int(box.cls[0]) in VEHICLE_CLASS_IDS:
            return True
    return False


def detectar_danos(ruta_imagen: str, conf_umbral: float = CONF_DEFAULT) -> tuple:
    """
    Ejecuta inferencia YOLO con validación de vehículo primero (COCO).

    Flujo:
        1. Verifica con YOLOv8n (COCO) si hay vehículo en la imagen.
           - Si no hay vehículo → lanza VehicleNotFoundError (rechaza perros, personas, etc.)
           - Si hay vehículo    → continúa
        2. Corre el modelo de daños especializado.
        3. Retorna imagen anotada + lista de detecciones (puede ser vacía si no hay daños).

    Args:
        ruta_imagen (str): Ruta al archivo de imagen (.jpg, .png, etc.)
        conf_umbral (float): Umbral mínimo de confianza para detección de daños (0.0 - 1.0)

    Returns:
        tuple: (imagen_anotada_rgb, lista_detecciones)

    Raises:
        FileNotFoundError: Si el modelo best.pt no existe.
        ValueError: Si la imagen no puede ser cargada.
        VehicleNotFoundError: Si no se detecta ningún vehículo en la imagen.
    """
    # ── Etapa 1: Validar vehículo con COCO ───────────────────────────────────
    if not validar_vehiculo(ruta_imagen):
        raise VehicleNotFoundError(
            "No se detectó un vehículo en la imagen. Por favor sube una fotografía de un automóvil, "
            "camioneta o autobús. Nota: Se requiere que al menos el 30% del volumen del vehículo esté "
            "visible dentro del encuadre para asegurar su reconocimiento."
        )

    # ── Etapa 2: Detectar daños con el modelo especializado ──────────────────
    modelo = cargar_modelo()

    img_path = Path(ruta_imagen)
    if not img_path.exists():
        raise ValueError(f"Imagen no encontrada: {ruta_imagen}")

    img_cv = cv2.imread(str(img_path))
    if img_cv is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")

    alto_img, ancho_img = img_cv.shape[:2]

    results = modelo.predict(
        source=str(ruta_imagen),
        conf=conf_umbral,
        verbose=False,
    )
    result = results[0]

    imagen_anotada_bgr = result.plot()
    imagen_anotada_rgb = cv2.cvtColor(imagen_anotada_bgr, cv2.COLOR_BGR2RGB)

    detecciones = []
    for box in result.boxes:
        clase_id  = int(box.cls[0])
        clase     = modelo.names[clase_id]
        confianza = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        zona = obtener_zona(x1, y1, x2, y2, ancho_img, alto_img)
        detecciones.append({
            "clase":         clase,
            "clase_legible": nombre_clase_legible(clase),
            "confianza":     round(confianza, 4),
            "zona":          zona,
            "coordenadas":   {
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
            },
        })

    detecciones.sort(key=lambda d: d["confianza"], reverse=True)
    return imagen_anotada_rgb, detecciones



def imagen_original_rgb(ruta_imagen: str) -> np.ndarray:
    """Lee y devuelve la imagen original en formato RGB para mostrar en Streamlit."""
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
