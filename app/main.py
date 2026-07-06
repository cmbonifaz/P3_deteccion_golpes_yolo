"""
main.py
-------
Interfaz de usuario Streamlit para el Sistema de Detección de Daños en Vehículos.

Ejecutar con:
    streamlit run app/main.py

Modo demo (sin modelo):
    streamlit run app/main.py -- --demo
    o activar el toggle en la barra lateral
"""

import os
import sys
import tempfile
import numpy as np
import streamlit as st
from pathlib import Path
import pandas as pd

# Agregar el directorio app/ al path para importar módulos locales
sys.path.insert(0, str(Path(__file__).parent))

from llm_report import generar_reporte
from utils import ESTADO_CONFIG, formatear_confianza, nombre_clase_legible, generar_texto_reporte

# ============================================================
# Configuración de la página
# ============================================================
st.set_page_config(
    page_title="Detección de Daños en Vehículos",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS personalizado para una UI más premium
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    .header-title {
        color: #f1f5f9; font-size: 2rem; font-weight: 700;
        margin: 0 0 0.5rem 0; letter-spacing: -0.5px;
    }
    .header-subtitle { color: #94a3b8; font-size: 1rem; margin: 0; }
    .header-badge {
        display: inline-block; background: #3b82f6; color: white;
        font-size: 0.75rem; font-weight: 600; padding: 0.25rem 0.75rem;
        border-radius: 100px; margin-bottom: 1rem; letter-spacing: 0.5px;
    }
    .demo-banner {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white; padding: 0.6rem 1.2rem; border-radius: 8px;
        font-weight: 600; font-size: 0.9rem; margin-bottom: 1.5rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .estado-badge {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.75rem 1.5rem; border-radius: 12px;
        font-size: 1.25rem; font-weight: 700; border: 2px solid;
        margin-bottom: 1rem;
    }
    .section-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Detecciones simuladas para modo DEMO
# ============================================================
DETECCIONES_DEMO = [
    {"clase": "bumper_damage",    "clase_legible": "Parachoques",    "confianza": 0.91, "zona": "Inferior Central"},
    {"clase": "headlight_damage", "clase_legible": "Faro delantero", "confianza": 0.87, "zona": "Inferior Central"},
    {"clase": "dent",             "clase_legible": "Abolladura",     "confianza": 0.78, "zona": "Lateral Derecho"},
    {"clase": "scratch",          "clase_legible": "Rayón",          "confianza": 0.65, "zona": "Lateral Derecho"},
    {"clase": "glass_crack",      "clase_legible": "Parabrisas dañado", "confianza": 0.83, "zona": "Zona Central"},
]


def imagen_demo_rgb() -> np.ndarray:
    """
    Genera una imagen placeholder para el modo demo.
    Devuelve un array RGB de 640x640 con un diseño de cuadrícula gris.
    """
    img = np.ones((480, 640, 3), dtype=np.uint8) * 45
    # Líneas de cuadrícula sutiles
    for i in range(0, 480, 60):
        img[i, :] = [60, 60, 60]
    for j in range(0, 640, 80):
        img[:, j] = [60, 60, 60]
    # Texto placeholder en el centro (como píxeles blancos)
    img[220:260, 270:370] = [80, 80, 80]
    return img


def imagen_demo_anotada() -> np.ndarray:
    """
    Genera una imagen con bounding boxes simuladas para el modo demo.
    """
    img = imagen_demo_rgb().copy()
    # Simular boxes como rectángulos con colores distintos
    colores = [(255, 100, 100), (255, 180, 0), (100, 200, 255), (100, 255, 150), (200, 100, 255)]
    boxes_demo = [
        (20, 320, 250, 460),    # bumper
        (260, 310, 390, 410),   # headlight
        (380, 200, 580, 350),   # dent
        (400, 140, 600, 200),   # scratch
        (180, 80, 480, 220),    # glass_crack
    ]
    for (x1, y1, x2, y2), color in zip(boxes_demo, colores):
        img[y1:y1+4, x1:x2] = color   # top
        img[y2-4:y2, x1:x2] = color   # bottom
        img[y1:y2, x1:x1+4] = color   # left
        img[y1:y2, x2-4:x2] = color   # right
    return img


# ============================================================
# Sidebar — modo demo
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    modelo_path = Path(__file__).parent.parent / "models" / "best.pt"
    modelo_listo = modelo_path.exists()

    if modelo_listo:
        st.success("✅ Modelo cargado")
        modo_demo = st.toggle("Modo Demo", value=False,
                              help="Usar detecciones simuladas en lugar del modelo real")
    else:
        st.warning("⚠️ Modelo no encontrado")
        st.caption("`models/best.pt` no existe aún.\nActivando modo demo automáticamente.")
        modo_demo = True

    st.divider()
    st.markdown("**Umbral de confianza**")
    conf_threshold = st.slider(
        "Umbral de confianza",
        0.1, 0.9, 0.4, 0.05,
        label_visibility="collapsed",
        help="Mínima confianza para mostrar una detección"
    )
    st.divider()
    st.markdown("**Sobre el sistema**")
    st.caption("YOLOv8s + Groq AI\n\n14 clases de daño vehicular\n\n~13,000 imágenes de entrenamiento")


# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="header-container">
    <div class="header-badge">🔬 YOLOv8 + Groq AI</div>
    <h1 class="header-title">🚗 Detección de Daños en Vehículos</h1>
    <p class="header-subtitle">
        Sube una foto del vehículo y el sistema detectará los daños automáticamente
        usando visión por computadora y generará un reporte explicado con IA.
    </p>
</div>
""", unsafe_allow_html=True)

# Banner de modo demo
if modo_demo:
    st.markdown("""
    <div class="demo-banner">
        🧪 MODO DEMO ACTIVO — Las detecciones son simuladas para ilustrar el flujo de la aplicación.
        Cuando el modelo esté listo, desactiva este modo desde la barra lateral.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Zona de carga de imagen
# ============================================================
st.markdown("### 📤 Cargar imagen del vehículo")

col_upload, col_info = st.columns([2, 1])

with col_upload:
    archivo_subido = st.file_uploader(
        label="Arrastra o selecciona una imagen",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Formatos soportados: JPG, JPEG, PNG, BMP, WEBP",
        label_visibility="collapsed",
    )
    if modo_demo and archivo_subido is None:
        usar_demo_directo = st.button(
            "🧪 Probar con detecciones de ejemplo (sin imagen)",
            use_container_width=True,
        )
    else:
        usar_demo_directo = False

with col_info:
    st.markdown("""
    <div class="section-card">
        <strong>💡 Recomendaciones para mejores resultados:</strong>
        <ul style="margin-top: 0.5rem; color: #475569; font-size: 0.9rem;">
            <li>Buena iluminación natural o artificial</li>
            <li>El vehículo debe ser el elemento principal</li>
            <li>Evita ángulos muy extremos (&lt;45°)</li>
            <li>Resolución mínima: 640×640 px</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Función de renderizado de resultados (compartida entre modos)
# ============================================================
def renderizar_resultados(img_original, img_anotada, detecciones, nombre_archivo="imagen"):
    """Renderiza imágenes, tabla de detecciones y sección de reporte LLM."""

    st.markdown("### 🖼️ Resultado de la Detección")
    col_orig, col_anotada = st.columns(2)
    with col_orig:
        st.markdown("**Imagen original**")
        st.image(img_original, use_container_width=True)
    with col_anotada:
        st.markdown("**Daños detectados**")
        st.image(img_anotada, use_container_width=True)

    st.divider()

    n_danos = len(detecciones)
    st.markdown(f"### 📋 Detecciones encontradas: **{n_danos}**")

    if n_danos == 0:
        st.success("✅ No se detectaron daños visibles con el umbral actual.")
        st.info("💡 Prueba reduciendo el umbral de confianza en la barra lateral.")
    else:
        df = pd.DataFrame([{
            "Tipo de Daño": d["clase_legible"],
            "Confianza":    formatear_confianza(d["confianza"]),
            "Zona":         d["zona"],
        } for d in detecciones])
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Reporte LLM
    st.markdown("### 🤖 Reporte con IA Explicativa")
    generar_btn = st.button(
        "✨ Generar reporte con Groq",
        type="primary",
        use_container_width=True,
        key="btn_reporte",
    )

    if generar_btn:
        try:
            with st.spinner("🧠 Analizando con Groq AI..."):
                reporte = generar_reporte(detecciones)

            estado = reporte.get("estado", "Desconocido")
            justificacion = reporte.get("justificacion", "Sin información.")
            config = ESTADO_CONFIG.get(estado, ESTADO_CONFIG["Sin daños"])

            st.markdown(
                f"""
                <div class="estado-badge" style="
                    color: {config['color']};
                    background-color: {config['bg']};
                    border-color: {config['color']};
                ">
                    {config['emoji']} Estado General del Vehículo: <strong>{estado}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Evaluación detallada:**")
            st.markdown(
                f"""
                <div class="section-card">
                    <p style="color: #334155; line-height: 1.7; margin: 0;">
                        {justificacion}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.divider()
            texto_reporte = generar_texto_reporte(estado, justificacion, detecciones)
            st.download_button(
                label="⬇️ Descargar reporte (.txt)",
                data=texto_reporte,
                file_name=f"reporte_vehiculo_{nombre_archivo}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"❌ Error al generar el reporte: {e}")


# ============================================================
# Lógica principal: imagen real o demo
# ============================================================
if archivo_subido is not None:
    st.divider()

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(archivo_subido.name).suffix
    ) as tmp:
        tmp.write(archivo_subido.read())
        ruta_temp = tmp.name

    try:
        if modo_demo:
            # Modo demo con imagen cargada por el usuario (solo muestra la imagen real)
            import cv2
            img_bgr = cv2.imread(ruta_temp)
            if img_bgr is None:
                st.error("❌ No se pudo leer la imagen.")
                st.stop()
            img_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_anotada = imagen_demo_anotada()  # boxes simuladas
            detecciones = DETECCIONES_DEMO

            st.info("🧪 Modo demo: se muestra tu imagen pero las detecciones son simuladas.")
        else:
            # Modo real: cargar modelo e inferir
            from inference import detectar_danos, imagen_original_rgb
            with st.spinner("🔍 Analizando imagen con YOLOv8..."):
                img_original = imagen_original_rgb(ruta_temp)
                img_anotada, detecciones = detectar_danos(ruta_temp, conf_umbral=conf_threshold)

        renderizar_resultados(
            img_original, img_anotada, detecciones,
            nombre_archivo=Path(archivo_subido.name).stem
        )

    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
    finally:
        try:
            os.unlink(ruta_temp)
        except Exception:
            pass

elif usar_demo_directo:
    # Demo sin imagen: usar imagen placeholder + detecciones simuladas
    st.divider()
    st.info("🧪 Modo demo sin imagen: mostrando detecciones simuladas sobre imagen de ejemplo.")
    renderizar_resultados(
        imagen_demo_rgb(),
        imagen_demo_anotada(),
        DETECCIONES_DEMO,
        nombre_archivo="demo",
    )

else:
    # Estado inicial — placeholder visual
    st.markdown("""
    <div style="
        text-align: center; padding: 4rem 2rem;
        background: #f8fafc; border: 2px dashed #cbd5e1;
        border-radius: 16px; color: #94a3b8;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
        <p style="font-size: 1.1rem; font-weight: 500; margin: 0;">
            Sube una imagen del vehículo para comenzar el análisis
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            El sistema detectará rayones, abolladuras, faros rotos, parachoques dañados y más.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("**🔬 Modelo:** YOLOv8s — 14 clases")
with col_f2:
    st.markdown("**🤖 LLM:** Gemini 2.0 Flash-Lite")
with col_f3:
    estado_modelo = "✅ Listo" if modelo_listo else "⏳ Entrenando..."
    st.markdown(f"**🎯 Estado:** {estado_modelo}")
