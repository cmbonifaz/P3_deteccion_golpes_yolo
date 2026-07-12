# 🛠️ Guía de Configuración de Supabase para Reportes de Vehículos

Esta guía detalla los pasos para crear la base de datos y el almacenamiento de archivos (Storage Bucket) en Supabase para persistir los reportes de inspección en la nube.

---

## 1. Crear la Tabla en PostgreSQL

En tu panel de Supabase, ve al **SQL Editor** y ejecuta la siguiente consulta para crear la tabla `reportes` con índices optimizados para las búsquedas que ya tienes implementadas (placa, estado, inspector, fecha):

```sql
-- Crear tabla de reportes
CREATE TABLE IF NOT EXISTS public.reportes (
    id TEXT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    fecha_hora TEXT NOT NULL,
    placa VARCHAR(20) NOT NULL,
    marca VARCHAR(100) NOT NULL,
    modelo VARCHAR(100) NOT NULL,
    inspector VARCHAR(100) NOT NULL,
    estado VARCHAR(20) NOT NULL,
    justificacion TEXT,
    pdf_nombre TEXT NOT NULL,
    pdf_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- NOTA: Para bases de datos ya existentes, ejecuta tambien:
-- ALTER TABLE public.reportes ADD COLUMN IF NOT EXISTS justificacion TEXT;

-- Habilitar Row Level Security (RLS)
ALTER TABLE public.reportes ENABLE ROW LEVEL SECURITY;

-- Crear políticas para permitir lectura/escritura pública (o según requieras)
CREATE POLICY "Permitir lectura pública de reportes" 
ON public.reportes FOR SELECT 
USING (true);

CREATE POLICY "Permitir inserción pública de reportes" 
ON public.reportes FOR INSERT 
WITH CHECK (true);

CREATE POLICY "Permitir eliminación pública de reportes" 
ON public.reportes FOR DELETE 
USING (true);

-- Crear índices para optimizar búsquedas frecuentes
CREATE INDEX IF NOT EXISTS idx_reportes_placa ON public.reportes (lower(placa));
CREATE INDEX IF NOT EXISTS idx_reportes_estado ON public.reportes (estado);
CREATE INDEX IF NOT EXISTS idx_reportes_inspector ON public.reportes (inspector);
CREATE INDEX IF NOT EXISTS idx_reportes_timestamp ON public.reportes (timestamp DESC);
```

---

## 2. Configurar el Almacenamiento (Storage Bucket)

Para guardar los archivos PDF, debes crear un **Storage Bucket** público:

1. Ve a la sección de **Storage** en tu barra lateral de Supabase.
2. Haz clic en **New Bucket**.
3. Nombra el bucket como: `reportes-inspeccion`.
4. Asegúrate de marcar la opción de **Public bucket** (para poder acceder y descargar los reportes por medio de URLs públicas).
5. Ve a **Policies** dentro de la sección de Storage y añade políticas de acceso (Policies) para el bucket `reportes-inspeccion`:
   - **Allowed operations:** Selecciona `Select` y `Insert`.
   - **Target roles:** `public` (o autenticado si prefieres restringir acceso).
   - O usa la plantilla rápida **"Give users access to read/write/update"** para acceso completo.

---

## 3. Variables de Entorno (.env)

Una vez creada tu base de datos y tu bucket, añade las credenciales de conexión en tu archivo `.env` en la raíz del proyecto:

```env
# Configuración de Supabase
SUPABASE_URL=https://tu-proyecto-id.supabase.co
SUPABASE_KEY=tu-anon-public-key
```

Puedes encontrar estas claves en tu panel de Supabase en **Settings ⚙️ > API**.
