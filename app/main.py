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

import cv2

# Inicializar estado para múltiples imágenes
if "imagenes_analizadas" not in st.session_state:
    st.session_state.imagenes_analizadas = []
if "ultimo_cam_bytes" not in st.session_state:
    st.session_state.ultimo_cam_bytes = None
if "videos_procesados" not in st.session_state:
    st.session_state.videos_procesados = []


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
st.markdown("### 📤 Cargar imagen(es) del vehículo")

col_upload, col_info = st.columns([2, 1])

with col_upload:
    tab_subir, tab_camara, tab_video = st.tabs(["📁 Subir Archivo(s)", "📷 Usar Cámara", "🎥 Subir Video"])
    
    archivos_subidos = None
    captura_camara = None
    archivo_video = None
    
    with tab_subir:
        archivos_subidos = st.file_uploader(
            label="Arrastra o selecciona una o más imágenes",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Formatos soportados: JPG, JPEG, PNG, BMP, WEBP (Máx. 10 fotos)",
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        
    with tab_camara:
        captura_camara = st.camera_input(
            "Toma una foto de los daños del vehículo",
            label_visibility="collapsed",
        )

    with tab_video:
        archivo_video = st.file_uploader(
            label="Arrastra o selecciona un archivo de video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Formatos soportados: MP4, MOV, AVI, MKV, WEBM (Muestreo de 1 frame/segundo)",
            label_visibility="collapsed",
        )
        
    # 1. Sincronizar archivos eliminados del uploader y video para actualizar el estado
    nombres_subidos = [f.name for f in archivos_subidos] if archivos_subidos else []
    nombre_video_actual = archivo_video.name if archivo_video else None
    
    st.session_state.imagenes_analizadas = [
        img for img in st.session_state.imagenes_analizadas
        if img["nombre"].startswith("Captura_Camara_") 
        or img["nombre"] in nombres_subidos
        or (nombre_video_actual and img["nombre"].startswith(nombre_video_actual))
    ]

    # Limpiar video no seleccionado
    if not nombre_video_actual:
        st.session_state.videos_procesados = []

    # 2. Procesar nuevos archivos subidos
    if archivos_subidos:
        # Calcular cuántos faltan por procesar reales
        nuevos_archivos = [f for f in archivos_subidos if not any(img["nombre"] == f.name for img in st.session_state.imagenes_analizadas)]
        
        if len(st.session_state.imagenes_analizadas) + len(nuevos_archivos) > 10:
            st.warning("⚠️ Límite de 10 imágenes excedido. Solo se procesarán las primeras 10.")
            limite_disp = 10 - len(st.session_state.imagenes_analizadas)
            nuevos_archivos = nuevos_archivos[:limite_disp]
            
        for archivo in nuevos_archivos:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(archivo.name).suffix or ".png") as tmp:
                tmp.write(archivo.read())
                ruta_temp = tmp.name
                
            try:
                if modo_demo:
                    import cv2
                    img_bgr = cv2.imread(ruta_temp)
                    if img_bgr is not None:
                        img_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        img_anotada = imagen_demo_anotada()
                        st.session_state.imagenes_analizadas.append({
                            "nombre": archivo.name,
                            "original": img_original,
                            "anotada": img_anotada,
                            "detecciones": DETECCIONES_DEMO
                        })
                else:
                    from inference import detectar_danos, imagen_original_rgb
                    with st.spinner(f"🔍 Analizando {archivo.name} con YOLOv8..."):
                        img_original = imagen_original_rgb(ruta_temp)
                        img_anotada, detecciones = detectar_danos(ruta_temp, conf_umbral=conf_threshold)
                        st.session_state.imagenes_analizadas.append({
                            "nombre": archivo.name,
                            "original": img_original,
                            "anotada": img_anotada,
                            "detecciones": detecciones
                        })
            except VehicleNotFoundError as e:
                st.warning(f"🚫 La imagen '{archivo.name}' fue descartada: {e}")
            except Exception as e:
                st.error(f"❌ Error al procesar '{archivo.name}': {e}")
            finally:
                try:
                    os.unlink(ruta_temp)
                except:
                    pass

    # 3. Procesar capturas de cámara
    if captura_camara is not None:
        bytes_actuales = captura_camara.getvalue()
        if st.session_state.ultimo_cam_bytes != bytes_actuales:
            if len(st.session_state.imagenes_analizadas) >= 10:
                st.warning("⚠️ Límite de 10 imágenes alcanzado. Limpia la inspección para agregar más.")
            else:
                cant_cam = len([x for x in st.session_state.imagenes_analizadas if x["nombre"].startswith("Captura_Camara_")])
                nombre_cam = f"Captura_Camara_{cant_cam + 1}.png"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(bytes_actuales)
                    ruta_temp = tmp.name
                    
                try:
                    if modo_demo:
                        import cv2
                        img_bgr = cv2.imread(ruta_temp)
                        if img_bgr is not None:
                            img_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            img_anotada = imagen_demo_anotada()
                            st.session_state.imagenes_analizadas.append({
                                "nombre": nombre_cam,
                                "original": img_original,
                                "anotada": img_anotada,
                                "detecciones": DETECCIONES_DEMO
                            })
                    else:
                        from inference import detectar_danos, imagen_original_rgb
                        with st.spinner("🔍 Analizando captura de cámara con YOLOv8..."):
                            img_original = imagen_original_rgb(ruta_temp)
                            img_anotada, detecciones = detectar_danos(ruta_temp, conf_umbral=conf_threshold)
                            st.session_state.imagenes_analizadas.append({
                                "nombre": nombre_cam,
                                "original": img_original,
                                "anotada": img_anotada,
                                "detecciones": detecciones
                            })
                    st.session_state.ultimo_cam_bytes = bytes_actuales
                except VehicleNotFoundError as e:
                    st.warning(f"🚫 Captura omitida: {e}")
                except Exception as e:
                    st.error(f"❌ Error al procesar captura: {e}")
                finally:
                    try:
                        os.unlink(ruta_temp)
                    except:
                        pass

    # 4. Procesar nuevo video subido
    if archivo_video is not None and nombre_video_actual not in st.session_state.videos_procesados:
        if len(st.session_state.imagenes_analizadas) >= 10:
            st.warning("⚠️ Límite de 10 imágenes alcanzado. Limpia la inspección para procesar el video.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(archivo_video.name).suffix or ".mp4") as tmp:
                tmp.write(archivo_video.read())
                ruta_video_temp = tmp.name
                
            try:
                cap = cv2.VideoCapture(ruta_video_temp)
                if not cap.isOpened():
                    st.error("❌ No se pudo abrir el archivo de video.")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0:
                        fps = 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Extraer exactamente 20 frames uniformemente distribuidos
                    num_frames_objetivo = 20
                    step = max(1.0, total_frames / num_frames_objetivo)
                    
                    frames_extraidos = []
                    for i in range(num_frames_objetivo):
                        frame_idx = int(i * step)
                        if frame_idx >= total_frames:
                            break
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        exito, frame = cap.read()
                        if not exito:
                            continue
                            
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        segundo = round(frame_idx / fps, 1)
                        frames_extraidos.append({
                            "segundo": segundo,
                            "imagen": frame_rgb
                        })
                        
                    cap.release()
                    
                    disp_libre = 10 - len(st.session_state.imagenes_analizadas)
                    
                    frames_procesados_con_exito = 0
                    frames_con_danos = 0
                    
                    for item in frames_extraidos:
                        # Si ya llenamos el cupo disponible (máximo 10 imágenes en total en la sesión), detenemos el análisis
                        if frames_con_danos >= disp_libre:
                            st.warning("⚠️ Se alcanzó el cupo máximo de 10 imágenes analizadas en la sesión. Se detuvo el análisis del video.")
                            break
                            
                        seg = item["segundo"]
                        img_rgb = item["imagen"]
                        nombre_frame = f"{nombre_video_actual} (Seg. {seg})"
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_frame:
                            cv2.imwrite(tmp_frame.name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                            ruta_frame_temp = tmp_frame.name
                            
                        try:
                            if modo_demo:
                                img_anotada = imagen_demo_anotada()
                                st.session_state.imagenes_analizadas.append({
                                    "nombre": nombre_frame,
                                    "original": img_rgb,
                                    "anotada": img_anotada,
                                    "detecciones": DETECCIONES_DEMO
                                })
                                frames_procesados_con_exito += 1
                                frames_con_danos += 1
                            else:
                                from inference import detectar_danos
                                with st.spinner(f"🔍 Analizando frame en segundo {seg}..."):
                                    img_anotada, detecciones = detectar_danos(ruta_frame_temp, conf_umbral=conf_threshold)
                                    
                                    # Solo guardar el frame si tiene detecciones de daños
                                    if len(detecciones) > 0:
                                        st.session_state.imagenes_analizadas.append({
                                            "nombre": nombre_frame,
                                            "original": img_rgb,
                                            "anotada": img_anotada,
                                            "detecciones": detecciones
                                        })
                                        frames_con_danos += 1
                                    
                                    frames_procesados_con_exito += 1
                        except VehicleNotFoundError:
                            pass
                        except Exception as e:
                            st.error(f"❌ Error al procesar el segundo {seg} del video: {e}")
                        finally:
                            try:
                                os.unlink(ruta_frame_temp)
                            except:
                                pass
                                
                    if frames_procesados_con_exito == 0:
                        st.warning("🚫 No se detectó ningún vehículo en ningún frame del video. Asegúrate de enfocar el automóvil.")
                    elif frames_con_danos == 0:
                        st.info("✅ Se analizó el video por completo y no se detectaron daños en ningún frame.")
                    else:
                        st.success(f"✅ Se analizaron los frames y se detectaron daños en {frames_con_danos} de ellos.")
                        
                    st.session_state.videos_procesados.append(nombre_video_actual)
            except Exception as e:
                st.error(f"❌ Error al decodificar video: {e}")
            finally:
                try:
                    os.unlink(ruta_video_temp)
                except:
                    pass

    # Usar demo directo sólo si no hay imágenes reales
    if modo_demo and len(st.session_state.imagenes_analizadas) == 0:
        usar_demo_directo = st.button(
            "🧪 Probar con detecciones de ejemplo (sin imagen)",
            use_container_width=True,
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
def renderizar_resultados(imagenes_analizadas: list):
    """Renderiza galería, tabla consolidada y el reporte LLM de múltiples imágenes."""

    st.markdown("### 🖼️ Registro Fotográfico de Inspección")
    
    # Selector de imagen para ver en detalle
    nombres_fotos = [img["nombre"] for img in imagenes_analizadas]
    
    col_sel, col_limpiar = st.columns([3, 1])
    with col_sel:
        foto_seleccionada = st.selectbox(
            "🔍 Selecciona una imagen para ver detalles de anotación:",
            nombres_fotos,
            label_visibility="collapsed"
        )
    with col_limpiar:
        if st.button("🧹 Limpiar Inspección", use_container_width=True, type="secondary"):
            st.session_state.imagenes_analizadas = []
            st.session_state.ultimo_cam_bytes = None
            st.rerun()
            
    # Encontrar la imagen seleccionada
    img_info = next(img for img in imagenes_analizadas if img["nombre"] == foto_seleccionada)
    
    col_orig, col_anotada = st.columns(2)
    with col_orig:
        st.markdown(f"**Fotografía Original ({img_info['nombre']})**")
        st.image(img_info["original"], use_container_width=True)
    with col_anotada:
        st.markdown("**Daños detectados (YOLO)**")
        st.image(img_info["anotada"], use_container_width=True)

    st.divider()

    # Consolidador de todas las detecciones de todas las imágenes
    todas_detecciones = []
    for img in imagenes_analizadas:
        for det in img["detecciones"]:
            det_con_origen = det.copy()
            det_con_origen["imagen"] = img["nombre"]
            todas_detecciones.append(det_con_origen)
            
    n_danos_totales = len(todas_detecciones)
    st.markdown(f"### 📋 Resumen de Daños Detectados (Total: **{n_danos_totales}**)")

    if n_danos_totales == 0:
        st.success("✅ No se detectaron daños visibles en ninguna de las imágenes analizadas.")
    else:
        df = pd.DataFrame([{
            "Foto/Origen":  d.get("imagen", "Imagen"),
            "Tipo de Daño": d["clase_legible"],
            "Confianza":    formatear_confianza(d["confianza"]),
            "Zona":         d["zona"],
        } for d in todas_detecciones])
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Reporte LLM consolidado
    st.markdown("### 🤖 Reporte con IA Explicativa (Consolidado)")
    generar_btn = st.button(
        "✨ Generar reporte consolidado con Groq",
        type="primary",
        use_container_width=True,
        key="btn_reporte",
    )

    if generar_btn:
        try:
            with st.spinner("🧠 Analizando todos los daños con Groq AI..."):
                reporte = generar_reporte(todas_detecciones)

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

            st.markdown("**Evaluación detallada de la IA:**")
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
            texto_reporte = generar_texto_reporte(estado, justificacion, todas_detecciones)
            
            try:
                pdf_bytes = bytes(generar_pdf_reporte(
                    imagenes_analizadas=imagenes_analizadas,
                    todas_detecciones=todas_detecciones,
                    estado=estado,
                    justificacion=justificacion,
                    identificador_vehiculo="Inspeccion_Multiple"
                ))
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="⬇️ Descargar Reporte PDF (Oficial)",
                        data=pdf_bytes,
                        file_name="reporte_inspeccion_consolidado.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                with col_d2:
                    st.download_button(
                        label="📄 Descargar Reporte TXT (Resumen)",
                        data=texto_reporte,
                        file_name="reporte_inspeccion_consolidado.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
            except Exception as pdf_err:
                st.warning(f"⚠️ No se pudo compilar el PDF: {pdf_err}. Ofreciendo versión de texto plano.")
                st.download_button(
                    label="⬇️ Descargar reporte (.txt)",
                    data=texto_reporte,
                    file_name="reporte_inspeccion_consolidado.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"❌ Error al generar el reporte: {e}")


# ============================================================
# Lógica principal de visualización
# ============================================================
if len(st.session_state.imagenes_analizadas) > 0:
    st.divider()
    renderizar_resultados(st.session_state.imagenes_analizadas)

elif usar_demo_directo:
    # Demo sin imagen: usar imagen placeholder + detecciones simuladas
    st.divider()
    st.info("🧪 Modo demo sin imagen: mostrando detecciones simuladas sobre imagen de ejemplo.")
    img_demo_info = [{
        "nombre": "demo_imagen.png",
        "original": imagen_demo_rgb(),
        "anotada": imagen_demo_anotada(),
        "detecciones": DETECCIONES_DEMO
    }]
    renderizar_resultados(img_demo_info)

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
            Sube imágenes del vehículo para comenzar el análisis
        </h3>
        <p style="font-size: 0.95rem; opacity: 0.75; margin-top: 0.75rem; max-width: 500px; margin-left: auto; margin-right: auto; line-height: 1.5;">
            El sistema de visión artificial detectará automáticamente abolladuras, rayones, faros rotos, parachoques dañados y más en múltiples capturas.
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
