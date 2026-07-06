"""
utils.py
--------
Funciones auxiliares compartidas por inference.py y main.py.
"""

from typing import Tuple

# ============================================================
# Colores por estado general del vehículo
# ============================================================
ESTADO_CONFIG = {
    "Leve": {
        "color": "#22c55e",       # verde
        "bg": "#dcfce7",
        "emoji": "🟢",
        "streamlit_color": "green",
    },
    "Moderado": {
        "color": "#f97316",       # naranja
        "bg": "#ffedd5",
        "emoji": "🟠",
        "streamlit_color": "orange",
    },
    "Grave": {
        "color": "#ef4444",       # rojo
        "bg": "#fee2e2",
        "emoji": "🔴",
        "streamlit_color": "red",
    },
    "Sin daños": {
        "color": "#6b7280",       # gris
        "bg": "#f3f4f6",
        "emoji": "✅",
        "streamlit_color": "gray",
    },
}


def obtener_zona(x1: float, y1: float, x2: float, y2: float,
                 ancho_img: int, alto_img: int) -> str:
    """
    Clasifica la ubicación aproximada de un daño en la imagen
    dividiendo la imagen en zonas.

    Zonas:
        - Superior 25% del alto → Techo
        - Inferior 25% del alto → Inferior (parachoques/faldón)
        - Tercio izquierdo → Lateral Izquierdo
        - Tercio derecho → Lateral Derecho
        - Centro → Frontal/Central
    """
    cx = (x1 + x2) / 2  # centro X de la caja
    cy = (y1 + y2) / 2  # centro Y de la caja

    # Zona vertical
    if cy < alto_img * 0.25:
        zona_v = "Techo"
    elif cy > alto_img * 0.75:
        zona_v = "Inferior"
    else:
        zona_v = "Cuerpo"

    # Zona horizontal
    if cx < ancho_img * 0.33:
        zona_h = "Izquierdo"
    elif cx > ancho_img * 0.67:
        zona_h = "Derecho"
    else:
        zona_h = "Central"

    # Combinar zonas
    if zona_v == "Techo":
        return "Techo"
    elif zona_v == "Inferior":
        return f"Inferior {zona_h}"
    else:
        return f"Lateral {zona_h}" if zona_h != "Central" else "Zona Central"


def formatear_confianza(conf: float) -> str:
    """Formatea la confianza como porcentaje con un decimal."""
    return f"{conf * 100:.1f}%"


def nombre_clase_legible(clase: str) -> str:
    """Convierte el nombre interno de clase a un texto legible en español."""
    TRADUCCIONES = {
        "scratch": "Rayón",
        "dent": "Abolladura",
        "crack": "Grieta de pintura",
        "glass_crack": "Parabrisas dañado",
        "headlight_damage": "Faro delantero",
        "taillight_damage": "Luz trasera",
        "mirror_damage": "Espejo lateral",
        "bumper_damage": "Parachoques",
        "door_damage": "Puerta",
        "fender_damage": "Guardafango",
        "hood_damage": "Cofre",
        "trunk_damage": "Cajuela",
        "roof_damage": "Techo",
        "rust": "Óxido",
    }
    return TRADUCCIONES.get(clase, clase.replace("_", " ").title())


def generar_texto_reporte(estado: str, justificacion: str,
                          detecciones: list) -> str:
    """
    Genera el texto del reporte para descargar como .txt
    """
    lineas = [
        "=" * 60,
        "  REPORTE DE INSPECCIÓN VEHICULAR",
        "  Sistema de Detección de Daños con IA",
        "=" * 60,
        "",
        f"ESTADO GENERAL: {estado}",
        "",
        "EVALUACIÓN:",
        justificacion,
        "",
        "-" * 60,
        "DETALLE DE DAÑOS DETECTADOS:",
        "",
    ]

    if not detecciones:
        lineas.append("  No se detectaron daños visibles.")
    else:
        for i, det in enumerate(detecciones, 1):
            lineas.append(
                f"  {i}. {nombre_clase_legible(det['clase'])} "
                f"— Confianza: {formatear_confianza(det['confianza'])} "
                f"— Zona: {det.get('zona', 'N/A')}"
            )

    lineas += [
        "",
        "=" * 60,
        "Generado automáticamente por el sistema YOLO + Gemini",
    ]
    return "\n".join(lineas)
