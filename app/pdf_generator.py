import os
import tempfile
import cv2
from datetime import datetime
from fpdf import FPDF

class ReportePDF(FPDF):
    def __init__(self, estado_general):
        super().__init__()
        self.estado_general = estado_general
        self.alias_nb_pages()

    def header(self):
        # Cabecera de página
        self.set_fill_color(15, 23, 42)  # Azul oscuro pizarra
        self.rect(0, 0, 210, 20, "F")
        
        # Línea decorativa neón inferior
        self.set_fill_color(99, 102, 241)  # Indigo
        self.rect(0, 20, 210, 1.5, "F")
        
        self.set_y(5)
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 11)
        self.cell(0, 10, "SISTEMA INTELIGENTE DE DETECCION DE DAÑOS VEHICULARES", align="C")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        
        # Línea divisoria sutil
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        
        self.cell(0, 10, f"Pagina {self.page_no()}/{{nb}} | Reporte Tecnico Oficial - Confidencial", align="L")
        self.cell(0, 10, "YOLOv8 + Groq AI", align="R")

def generar_pdf_reporte(img_original, img_anotada, detecciones, estado, justificacion, nombre_archivo="vehiculo"):
    """
    Genera un informe PDF profesional con imágenes, tablas y análisis.
    Devuelve los bytes del PDF generado.
    """
    # Crear PDF
    pdf = ReportePDF(estado)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # 1. TÍTULO Y FECHA
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "Reporte Tecnico de Inspeccion", ln=True)
    
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    fecha_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    pdf.cell(0, 5, f"Fecha de emision: {fecha_str} | ID Archivo: {nombre_archivo}", ln=True)
    pdf.ln(5)
    
    # 2. ESTADO DEL VEHÍCULO (Badge grande)
    colores_estado = {
        "Sin daños": {"text": (21, 128, 61), "bg": (240, 253, 244), "border": (134, 239, 172)},
        "Leve": {"text": (161, 98, 7), "bg": (254, 252, 232), "border": (253, 224, 71)},
        "Moderado": {"text": (194, 65, 12), "bg": (255, 247, 237), "border": (253, 186, 116)},
        "Grave": {"text": (185, 28, 28), "bg": (254, 242, 242), "border": (252, 165, 165)}
    }
    col = colores_estado.get(estado, colores_estado["Sin daños"])
    
    pdf.set_fill_color(*col["bg"])
    pdf.set_draw_color(*col["border"])
    pdf.set_text_color(*col["text"])
    pdf.set_font("helvetica", "B", 12)
    
    # Dibujar badge
    pdf.cell(190, 12, f"  ESTADO GENERAL DEL VEHICULO: {estado.upper()}", border=1, ln=True, fill=True)
    pdf.ln(5)
    
    # 3. JUSTIFICACIÓN DE LA IA
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "Analisis y Diagnostico de IA:", ln=True)
    
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 5, justificacion)
    pdf.ln(5)
    
    # 4. TABLA DE DETECCIONES
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "Detalle de Anomalidades Detectadas:", ln=True)
    
    # Cabecera de Tabla
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_text_color(71, 85, 105)
    pdf.set_font("helvetica", "B", 9)
    
    pdf.cell(80, 8, " Tipo de Daño", border=1, fill=True)
    pdf.cell(50, 8, " Zona Afectada", border=1, fill=True)
    pdf.cell(60, 8, " Confianza del Modelo", border=1, fill=True, ln=True)
    
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    
    if not detecciones:
        pdf.cell(190, 8, " No se detectaron daños en la carroceria.", border=1, ln=True, align="C")
    else:
        for d in detecciones:
            pdf.cell(80, 8, f" {d['clase_legible']}", border=1)
            pdf.cell(50, 8, f" {d['zona']}", border=1)
            pdf.cell(60, 8, f" {int(d['confianza'] * 100)}%", border=1, ln=True)
            
    pdf.ln(10)
    
    # 5. IMÁGENES DEL ANÁLISIS
    # Escribir imágenes en archivos temporales para cargarlas al PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_anot:
        
        # Guardar imágenes como archivos
        cv2.imwrite(tmp_orig.name, cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tmp_anot.name, cv2.cvtColor(img_anotada, cv2.COLOR_RGB2BGR))
        
        # Nueva página para las imágenes si es necesario, o al final
        pdf.add_page()
        
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, "Registro Fotografico del Analisis", ln=True)
        pdf.ln(2)
        
        # Posicionar imágenes una al lado de la otra (o en vertical si son anchas)
        # Ancho disponible 190 mm. Haremos dos de 90 mm cada una.
        y_pos = pdf.get_y()
        pdf.image(tmp_orig.name, x=10, y=y_pos, w=90, h=65)
        pdf.image(tmp_anot.name, x=110, y=y_pos, w=90, h=65)
        
        pdf.set_y(y_pos + 68)
        pdf.set_font("helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(90, 5, "Fotografia Original del Vehiculo", align="C")
        pdf.cell(110, 5, "Captura Analizada (Bounding Boxes YOLOv8)", align="C", ln=True)
        
        # Eliminar archivos temporales
        orig_name = tmp_orig.name
        anot_name = tmp_anot.name
        
    try:
        os.unlink(orig_name)
        os.unlink(anot_name)
    except Exception:
        pass
        
    # Obtener el contenido en bytes del PDF
    return bytes(pdf.output())
