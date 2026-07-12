# -*- coding: utf-8 -*-
"""
historial.py
------------
Módulo para la persistencia híbrida de reportes de inspección.
Guarda los reportes de manera local (JSON + PDF) y en la nube (Supabase PostgreSQL + Storage).
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Cargar variables de entorno
load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "reportes"
INDEX_FILE = DATA_DIR / "historial.json"
BUCKET_NAME = "reportes-inspeccion"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

_client = None

def obtener_cliente_supabase() -> Client:
    """Retorna el cliente de Supabase si está configurado y es accesible."""
    global _client
    if _client is not None:
        return _client
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            _client = create_client(SUPABASE_URL, SUPABASE_KEY)
            return _client
        except Exception as e:
            print(f"Advertencia: No se pudo conectar a Supabase: {e}")
    return None

def inicializar_directorios():
    """Asegura la existencia de la carpeta de datos e historial local."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def obtener_historial() -> list:
    """Retorna la lista de reportes históricos ordenados por fecha desc (más recientes primero)."""
    inicializar_directorios()
    
    # 1. Intentar con Supabase
    client = obtener_cliente_supabase()
    if client:
        try:
            res = client.table("reportes").select("*").order("timestamp", desc=True).execute()
            if res.data is not None:
                return res.data
        except Exception as e:
            print(f"Advertencia: Falló consulta a Supabase. Usando local. Detalle: {e}")
            
    # 2. Fallback a Local
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            historial = json.load(f)
        return sorted(historial, key=lambda x: x.get("timestamp", 0), reverse=True)
    except Exception as e:
        print(f"Error al leer historial local: {e}")
        return []

def guardar_en_historial(pdf_bytes: bytes, placa: str, marca: str, modelo: str, inspector: str, estado: str, justificacion: str = "") -> str:
    """
    Guarda el PDF localmente y en Supabase, y registra la entrada en el historial.
    Retorna el nombre del archivo PDF generado.
    """
    inicializar_directorios()
    
    timestamp_val = int(datetime.now().timestamp())
    fecha_hora_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    fecha_archivo_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Formatear datos limpios
    placa_limpia = placa.strip().upper() if placa and placa.strip() else "S-N"
    marca_limpia = marca.strip() if marca and marca.strip() else "S-D"
    modelo_limpia = modelo.strip() if modelo and modelo.strip() else "S-D"
    inspector_limpio = inspector.strip() if inspector and inspector.strip() else "S-D"
    
    # Reemplazar caracteres no permitidos en nombres de archivo
    placa_archivo = re.sub(r'[^a-zA-Z0-9_-]', '', placa_limpia)
    if not placa_archivo:
        placa_archivo = "SN"
        
    pdf_nombre = f"reporte_{placa_archivo}_{fecha_archivo_str}.pdf"
    pdf_path = DATA_DIR / pdf_nombre
    
    # 1. Guardar archivo PDF localmente
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    # Registrar entrada en JSON local
    nuevo_registro = {
        "id": f"{timestamp_val}_{placa_archivo}",
        "timestamp": timestamp_val,
        "fecha_hora": fecha_hora_str,
        "placa": placa_limpia,
        "marca": marca_limpia,
        "modelo": modelo_limpia,
        "inspector": inspector_limpio,
        "estado": estado,
        "justificacion": justificacion.strip() if justificacion else "",
        "pdf_nombre": pdf_nombre,
        "pdf_url": None
    }
    
    # 2. Intentar subir a Supabase
    client = obtener_cliente_supabase()
    if client:
        try:
            # Subir PDF a Storage
            client.storage.from_(BUCKET_NAME).upload(
                path=pdf_nombre,
                file=pdf_bytes,
                file_options={"content-type": "application/pdf"}
            )
            # Obtener URL pública
            pdf_url = client.storage.from_(BUCKET_NAME).get_public_url(pdf_nombre)
            nuevo_registro["pdf_url"] = pdf_url
            
            # Insertar registro en tabla
            client.table("reportes").insert(nuevo_registro).execute()
        except Exception as e:
            print(f"Advertencia: Falló sincronización con Supabase. Detalle: {e}")
            
    # 3. Actualizar índice JSON local
    historial = obtener_historial_local_exclusivo()
    historial.append(nuevo_registro)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)
        
    return pdf_nombre

def obtener_historial_local_exclusivo() -> list:
    """Lee exclusivamente el JSON local, útil para sincronización interna."""
    inicializar_directorios()
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def obtener_pdf_historial(pdf_nombre: str) -> bytes:
    """Lee y retorna los bytes del PDF histórico (intenta desde Supabase, luego local)."""
    # 1. Intentar con Supabase
    client = obtener_cliente_supabase()
    if client:
        try:
            pdf_bytes = client.storage.from_(BUCKET_NAME).download(pdf_nombre)
            if pdf_bytes:
                return pdf_bytes
        except Exception as e:
            print(f"Advertencia: No se pudo descargar PDF de Supabase: {e}. Intentando local.")
            
    # 2. Fallback a Local
    pdf_path = DATA_DIR / pdf_nombre
    if pdf_path.exists():
        try:
            with open(pdf_path, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"Error al leer PDF local: {e}")
    return b""

def eliminar_de_historial(reporte_id: str) -> bool:
    """
    Elimina un reporte por ID del historial (de Supabase y del almacenamiento local).
    Retorna True si la eliminación fue exitosa (al menos localmente o en la nube).
    """
    exito_supabase = False
    exito_local = False
    
    # 1. Intentar eliminar en Supabase
    client = obtener_cliente_supabase()
    if client:
        try:
            # Obtener el registro para conocer el nombre del PDF
            res = client.table("reportes").select("pdf_nombre").eq("id", reporte_id).execute()
            if res.data and len(res.data) > 0:
                pdf_nombre = res.data[0].get("pdf_nombre")
                if pdf_nombre:
                    try:
                        client.storage.from_(BUCKET_NAME).remove([pdf_nombre])
                    except Exception as st_err:
                        print(f"Advertencia: No se pudo borrar el PDF de Supabase Storage: {st_err}")
                
                # Borrar fila de base de datos
                client.table("reportes").delete().eq("id", reporte_id).execute()
                exito_supabase = True
        except Exception as e:
            print(f"Advertencia: Falló eliminación en Supabase: {e}")
            
    # 2. Eliminar en Local
    inicializar_directorios()
    try:
        historial = obtener_historial_local_exclusivo()
        nuevo_historial = []
        registro_a_eliminar = None
        
        for reg in historial:
            if reg.get("id") == reporte_id:
                registro_a_eliminar = reg
            else:
                nuevo_historial.append(reg)
                
        if registro_a_eliminar:
            # Borrar archivo PDF local
            pdf_nombre = registro_a_eliminar.get("pdf_nombre")
            if pdf_nombre:
                pdf_path = DATA_DIR / pdf_nombre
                if pdf_path.exists():
                    try:
                        os.remove(pdf_path)
                    except Exception as os_err:
                        print(f"Advertencia al borrar PDF local: {os_err}")
            
            # Guardar el JSON actualizado
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(nuevo_historial, f, ensure_ascii=False, indent=2)
            exito_local = True
    except Exception as e:
        print(f"Error al eliminar reporte localmente: {e}")
        
    return exito_supabase or exito_local
