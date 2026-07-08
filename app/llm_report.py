"""
llm_report.py
-------------
Capa de IA explicativa usando Groq (Llama 3).

Recibe la lista de detecciones de YOLO y genera:
    - Estado general: "Leve" | "Moderado" | "Grave" | "Sin daños"
    - Justificación: texto explicativo en español (máx. 100 palabras)

Usa la librería oficial 'groq' para garantizar alta velocidad y disponibilidad.
Enforce JSON format nativamente mediante response_format.
"""

import os
import json
import re
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ============================================================
# Configuración del cliente Groq
# ============================================================
_api_key       = os.getenv("GROQ_API_KEY")
_modelo_nombre = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Crear cliente una sola vez
if _api_key and _api_key.strip() not in ("", "tu_api_key_aqui"):
    _client = Groq(api_key=_api_key)
else:
    _client = None


def _preparar_detecciones_para_prompt(detecciones: list) -> list:
    """
    Simplifica las detecciones para el prompt del LLM:
    elimina coordenadas brutas y conserva solo clase, confianza, zona y origen de imagen.
    """
    resultado = []
    for d in detecciones:
        item = {
            "tipo_daño": d["clase_legible"],
            "confianza": f"{d['confianza'] * 100:.1f}%",
            "zona":      d["zona"],
        }
        if "imagen" in d:
            item["foto_origen"] = d["imagen"]
        resultado.append(item)
    return resultado


def _parsear_respuesta_json(texto: str) -> dict:
    """
    Parsea el JSON retornado por Groq.
    Si falla, intenta deducir el estado por fallback.
    """
    try:
        datos = json.loads(texto)
        estado = datos.get("estado", "Desconocido")
        justificacion = datos.get("justificacion", texto)
        
        # Si por alguna razón el LLM usó acento en la clave ("justificación")
        if "justificacion" not in datos and "justificación" in datos:
            justificacion = datos["justificación"]
            
        if estado not in ("Leve", "Moderado", "Grave", "Sin daños"):
            estado = _inferir_estado_fallback(justificacion)
            
        return {"estado": estado, "justificacion": justificacion}
    except json.JSONDecodeError:
        pass

    # Si por alguna razón falló el parsing directo de JSON completo
    patron_json = re.search(r"\{[\s\S]*\}", texto)
    if patron_json:
        try:
            datos = json.loads(patron_json.group(0))
            estado = datos.get("estado", "Desconocido")
            justificacion = datos.get("justificacion", datos.get("justificación", texto))
            if estado not in ("Leve", "Moderado", "Grave", "Sin daños"):
                estado = _inferir_estado_fallback(justificacion)
            return {"estado": estado, "justificacion": justificacion}
        except json.JSONDecodeError:
            pass

    return {
        "estado":       _inferir_estado_fallback(texto),
        "justificacion": texto,
    }


def _inferir_estado_fallback(texto: str) -> str:
    """Deduce el estado desde texto libre si el parsing JSON falla."""
    t = texto.lower()
    if any(p in t for p in ["grave", "severo", "crítico", "mayor", "estructural"]):
        return "Grave"
    if any(p in t for p in ["moderado", "medio", "considerable", "significativo"]):
        return "Moderado"
    if any(p in t for p in ["leve", "menor", "mínimo", "ligero", "superficial"]):
        return "Leve"
    if any(p in t for p in ["sin daño", "no se detectaron", "no hay", "ningún"]):
        return "Sin daños"
    return "Moderado"


def generar_reporte(detecciones: list, reintentos: int = 3) -> dict:
    """
    Genera un reporte explicativo del estado general del vehículo
    usando Groq.

    Args:
        detecciones (list): Lista de dicts con detecciones de YOLO.
        reintentos  (int):  Número de intentos ante errores de red o quota.

    Returns:
        dict: {
            "estado": "Leve" | "Moderado" | "Grave" | "Sin daños",
            "justificacion": str  (texto en español, máx. 100 palabras)
        }

    Raises:
        RuntimeError: Si la API key no está configurada.
    """
    if _client is None:
        raise RuntimeError(
            "La API key de Groq no está configurada.\n"
            "Crea el archivo .env con: GROQ_API_KEY=tu_key\n"
            "Obtén tu key en: https://console.groq.com/"
        )

    # Caso especial: lista vacía → sin daños (no gastar quota)
    if not detecciones:
        return {
            "estado": "Sin daños",
            "justificacion": (
                "No se detectaron daños visibles en el vehículo. "
                "La imagen no muestra evidencia de daños que el modelo "
                "pueda identificar con el umbral de confianza actual."
            ),
        }

    detecciones_simplificadas = _preparar_detecciones_para_prompt(detecciones)
    n_danos = len(detecciones)

    prompt = f"""
Eres un asistente experto en inspección y peritaje vehicular. Recibirás una lista de daños
detectados automáticamente por un modelo de visión por computadora (YOLO).

Tu tarea es generar un informe detallado en formato JSON estructurado.

1. Determinar el estado general del vehículo usando EXACTAMENTE una de estas categorías:
   - "Sin daños": ningún daño detectado (lista vacía)
   - "Leve": daños superficiales menores (como un rayón o una abolladura menor), sin comprometer la seguridad vial o la conducción
   - "Moderado": múltiples daños de tamaño medio o daños en partes de la carrocería (parachoques, puertas, guardabarros, capó) sin riesgo inmediato para la conducción
   - "Grave": daños estructurales severos, o si hay daños concurrentes en componentes de seguridad crítica (por ejemplo, vidrios o espejos agrietados Y luces rotas a la vez), o si el total de daños detectados es de 5 o más.

2. Redactar una justificación explicativa y descriptiva (entre 100 y 250 palabras) en español.
   DEBES enumerar, describir e indicar la localización de cada uno de los daños detectados en la lista (por ejemplo, "un rayón en la puerta lateral", "una abolladura en el guardabarros", etc.). 
   Explica claramente qué piezas están afectadas y cómo la combinación de estos daños justifica la categoría de estado asignada. Evita frases genéricas o cortas como "se superó el límite".

REGLAS ESTRICTAS:
- Basa tu análisis ÚNICAMENTE en los daños de la lista. Describe las piezas afectadas y su ubicación de acuerdo a los datos provistos.
- Si hay daños concurrentes en piezas críticas de seguridad (faros, parabrisas, espejos) o 5 o más daños totales, cataloga obligatoriamente el estado como "Grave".
- Sé profesional, técnico y detallado en la descripción de cada daño.
- No inventes marcas, modelos ni valores monetarios de reparación.
- Responde obligatoriamente en formato JSON válido con las llaves "estado" y "justificacion".

Número de daños detectados: {n_danos}
Daños detectados (JSON):
{json.dumps(detecciones_simplificadas, ensure_ascii=False, indent=2)}
"""

    espera = 5  # segundos de espera inicial ante rate limit / sobrecarga

    for intento in range(1, reintentos + 1):
        try:
            completion = _client.chat.completions.create(
                model=_modelo_nombre,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un sistema automático que genera reportes de inspección de vehículos únicamente en formato JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            
            respuesta_texto = completion.choices[0].message.content
            return _parsear_respuesta_json(respuesta_texto)

        except Exception as e:
            mensaje = str(e)

            # Si es rate limit (429) o sobrecarga (503/500) -> reintentar con backoff
            if any(err in mensaje for err in ["429", "503", "500", "Rate limit"]) and intento < reintentos:
                print(f"   [INFO] Error temporal de Groq. Reintentando en {espera}s (intento {intento+1}/{reintentos})...")
                time.sleep(espera)
                espera *= 2
                continue

            # Otros errores -> devolver mensaje amigable
            return {
                "estado": "Desconocido",
                "justificacion": (
                    f"Error al consultar el modelo de IA Groq: {mensaje}. "
                    "Verifica tu API key y conexión a internet."
                ),
            }

    return {
        "estado": "Desconocido",
        "justificacion": (
            "Se agotaron los intentos de conexión con la API de Groq (rate limit). "
            "Espera unos segundos y vuelve a intentarlo."
        ),
    }
