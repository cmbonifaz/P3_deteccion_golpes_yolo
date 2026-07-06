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
from pdf_generator import generar_pdf_reporte

# Importar excepción personalizada de inferencia (disponible aunque el modelo no esté cargado)
try:
    from inference import VehicleNotFoundError
except Exception:
    # Fallback por si inference falla al importar (modelo no instalado, etc.)
    class VehicleNotFoundError(Exception):  # type: ignore
        pass

# ============================================================
# Configuración de la página
# ============================================================
st.set_page_config(
    page_title="Detección de Daños en Vehículos",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS personalizado para una UI más premium
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
    }

    /* Ocultar barra superior por defecto de Streamlit y botones nativos */
    header { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    .stDeployButton { display: none !important; }

    /* Ajustar padding superior */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Estilo del Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Contenedor de Cabecera con diseño Premium Glass-Indigo y brillo */
    .header-container {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(30, 27, 75, 0.7) 100%);
        padding: 2.5rem 3rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(99, 102, 241, 0.25);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.12);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #3b82f6, #6366f1, #ec4899);
    }

    .header-title {
        color: #ffffff; 
        font-size: 2.6rem; 
        font-weight: 800;
        margin: 0.5rem 0 0.75rem 0; 
        letter-spacing: -1px;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle { 
        color: #94a3b8; 
        font-size: 1.1rem; 
        margin: 0;
        line-height: 1.6;
        max-width: 800px;
    }
    
    .header-badge {
        display: inline-flex; 
        align-items: center;
        background: rgba(99, 102, 241, 0.15); 
        color: #a5b4fc;
        font-size: 0.8rem; 
        font-weight: 600; 
        padding: 0.35rem 1rem;
        border-radius: 100px; 
        border: 1px solid rgba(99, 102, 241, 0.3);
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1);
    }

    /* Banner de Modo Demo */
    .demo-banner {
        background: linear-gradient(90deg, #d97706, #b45309);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fef3c7; 
        padding: 0.75rem 1.25rem; 
        border-radius: 12px;
        font-weight: 600; 
        font-size: 0.92rem; 
        margin-bottom: 2rem;
        display: flex; 
        align-items: center; 
        gap: 0.6rem;
        box-shadow: 0 4px 20px rgba(217, 119, 6, 0.1);
    }

    /* Insignias de Estado General */
    .estado-badge {
        display: inline-flex; 
        align-items: center; 
        gap: 0.75rem;
        padding: 1rem 2rem; 
        border-radius: 16px;
        font-size: 1.4rem; 
        font-weight: 800; 
        border: 2px solid;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }

    /* Tarjetas Glassmorphism adaptativas */
    .section-card {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px; 
        padding: 1.75rem; 
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    
    .section-card strong {
        color: #6366f1;
    }

    /* Custom Styling para el DataFrame */
    .stDataFrame { 
        border-radius: 12px; 
        overflow: hidden; 
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Sobrescribir botones globales de Streamlit */
    .stButton button, .stDownloadButton button {
        background: linear-gradient(95deg, #4f46e5 0%, #6366f1 50%, #ec4899 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        height: auto !important;
        display: block !important;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4), 0 0 10px rgba(236, 72, 153, 0.2) !important;
    }

    .stButton button:active, .stDownloadButton button:active {
        transform: translateY(1px) !important;
    }

    /* Estilo específico para botones de descarga */
    .stDownloadButton button {
        background: linear-gradient(95deg, #059669 0%, #10b981 100%) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2) !important;
    }

    .stDownloadButton button:hover {
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35) !important;
    }

    /* Estilo del File Uploader drag & drop */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #ec4899 !important;
        background: rgba(30, 41, 59, 0.5) !important;
    }

    /* Ocultar elementos extra de carga */
    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }
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
    foto_detalle = st.toggle(
        "🔍 Foto de detalle / close-up",
        value=False,
        help="¿La foto es de una parte del vehículo (puerta, capó, parachoques) "
             "sin mostrar el auto completo? Actívalo para saltarse la validación automática.",
    )
    if modo_demo and archivo_subido is None:
        usar_demo_directo = st.button(
            "🧪 Probar con detecciones de ejemplo (sin imagen)",
            width="stretch",
        )
    else:
        usar_demo_directo = False

with col_info:
    st.markdown("""
    <div class="section-card">
        <strong style="color: #3b82f6; font-size: 1.1rem; display: block; margin-bottom: 0.75rem;">💡 Recomendaciones para mejores resultados:</strong>
        <ul style="margin-top: 0rem; font-size: 0.92rem; line-height: 1.7; padding-left: 1.2rem; opacity: 0.85;">
            <li>Buena iluminación natural o artificial.</li>
            <li>El vehículo debe ser el elemento principal de la toma.</li>
            <li>Evita ángulos muy extremos (menores a 45°).</li>
            <li>Resolución de imagen mínima recomendada: 640×640 px.</li>
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
        st.image(img_original, width="stretch")
    with col_anotada:
        st.markdown("**Daños detectados**")
        st.image(img_anotada, width="stretch")

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
        st.dataframe(df, width="stretch", hide_index=True)

    st.divider()

    # Reporte LLM
    st.markdown("### 🤖 Reporte con IA Explicativa")
    generar_btn = st.button(
        "✨ Generar reporte con Groq",
        type="primary",
        width="stretch",
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
                    <p style="line-height: 1.7; margin: 0; color: var(--text-color); opacity: 0.95;">
                        {justificacion}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.divider()
            texto_reporte = generar_texto_reporte(estado, justificacion, detecciones)
            
            try:
                # Generar PDF en memoria
                pdf_bytes = bytes(generar_pdf_reporte(
                    img_original, img_anotada, detecciones,
                    estado, justificacion, nombre_archivo=nombre_archivo
                ))
                
                # Crear dos columnas para los botones de descarga
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="⬇️ Descargar Reporte PDF (Oficial)",
                        data=pdf_bytes,
                        file_name=f"reporte_inspeccion_{nombre_archivo}.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )
                with col_d2:
                    st.download_button(
                        label="📄 Descargar Reporte TXT (Resumen)",
                        data=texto_reporte,
                        file_name=f"reporte_vehiculo_{nombre_archivo}.txt",
                        mime="text/plain",
                        width="stretch",
                    )
            except Exception as pdf_err:
                st.warning(f"⚠️ No se pudo compilar el PDF: {pdf_err}. Ofreciendo versión de texto plano.")
                st.download_button(
                    label="⬇️ Descargar reporte (.txt)",
                    data=texto_reporte,
                    file_name=f"reporte_vehiculo_{nombre_archivo}.txt",
                    mime="text/plain",
                    width="stretch",
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
                img_anotada, detecciones = detectar_danos(
                    ruta_temp,
                    conf_umbral=conf_threshold,
                    skip_vehicle_check=foto_detalle,
                )

        renderizar_resultados(
            img_original, img_anotada, detecciones,
            nombre_archivo=Path(archivo_subido.name).stem
        )

    except VehicleNotFoundError as e:
        st.warning(f"🚫 {e}")
        st.info("💡 Si es una foto de detalle (puerta, capot, parachoques), activa el toggle **🔍 Foto de detalle / close-up** en la barra lateral.")
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
        text-align: center; padding: 5rem 2rem;
        background: rgba(128, 128, 128, 0.05);
        border: 2px dashed rgba(128, 128, 128, 0.2);
        border-radius: 20px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.05);
        backdrop-filter: blur(8px);
    ">
        <div style="font-size: 4.5rem; margin-bottom: 1.2rem; filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.3));">📷</div>
        <h3 style="font-size: 1.35rem; font-weight: 700; margin: 0; color: var(--text-color);">
            Sube una imagen del vehículo para comenzar el análisis
        </h3>
        <p style="font-size: 0.95rem; opacity: 0.75; margin-top: 0.75rem; max-width: 500px; margin-left: auto; margin-right: auto; line-height: 1.5;">
            El sistema de visión artificial detectará automáticamente abolladuras, rayones, faros rotos, parachoques dañados, óxido y más.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 1.75rem;
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 14px;
        font-size: 0.88rem;
        opacity: 0.9;
        color: var(--text-color);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    ">
        <div>🔬 <strong>Modelo:</strong> YOLOv8s (14 clases)</div>
        <div>🤖 <strong>LLM:</strong> Groq Llama-3.3-70b-versatile</div>
        <div>🎯 <strong>Estado del Sistema:</strong> {"🟢 Listo para inspección" if modelo_listo else "🟡 Modo Demo Activo (Cargando best.pt...)"}</div>
    </div>
    """,
    unsafe_allow_html=True
)
