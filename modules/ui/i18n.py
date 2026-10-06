"""Small, dependency-free UI catalog. English source strings remain stable keys.

Add translations here as screens are migrated; UI code and persisted values keep
their original English identifiers. This also lets old config files load.
"""
from __future__ import annotations

import re

from PySide6.QtWidgets import (QAbstractButton, QComboBox, QGroupBox, QLabel,
                               QLineEdit, QTabWidget, QTextEdit, QWidget)


ES = {
    "0/0": "0/0",
    "Duration: 300s (5:00)": "Duración: 300 s (5:00)",
    "Engine:": "Motor:",
    "Interval:": "Intervalo:",
    "Mention Pro when I reach something only it can do": "Avisarme cuando una función necesite Pro",
    "Search": "Buscar",
    "Send": "Enviar",
    "Stop": "Detener",
    "Stop on find": "Detener al encontrar",
    "Top-K:": "Mejores K:",
    "VideoHighlighter Pro": "VideoHighlighter Pro",
    "WARNING: Context is large! Small models (3B) may ignore parts. Use 8B+ models for best results.": "Advertencia: hay muchos datos. Los modelos pequeños (3B) podrían omitir parte del contexto; usa uno de 8B o más para mejores resultados.",
    "(select GGUF file below)": "(selecciona un archivo GGUF abajo)",
    "Auto (your action head if installed, else the old models)": "Automático (modelo de acciones entrenado o modelos anteriores)",
    "CLIP + LLM": "CLIP + IA",
    "CLIP only": "Solo CLIP",
    "Crop them out (experimental)": "Recortarlas (experimental)",
    "LLM only": "Solo IA",
    "Large (accurate, slower)": "Grande (preciso, más lento)",
    "Medium (balanced)": "Mediano (equilibrado)",
    "Nano (fastest, lowest accuracy)": "Nano (más rápido, menos preciso)",
    "OpenVINO, Intel model (deprecated)": "OpenVINO, modelo Intel (obsoleto)",
    "R3D + CPU (deprecated)": "R3D + CPU (obsoleto)",
    "R3D + CUDA (deprecated)": "R3D + CUDA (obsoleto)",
    "R3D + DirectML (deprecated)": "R3D + DirectML (obsoleto)",
    "R3D-18 (fastest)": "R3D-18 (más rápido)",
    "R(2+1)D-18 (most accurate)": "R(2+1)D-18 (más preciso)",
    "SigLIP2 + your trained action head": "SigLIP2 + modelo de acciones entrenado",
    "Skip those moments": "Excluir esos momentos",
    "Small (fast, good balance)": "Pequeño (rápido y equilibrado)",
    "Standard (80 objects)": "Estándar (80 objetos)",
    "Standard YOLOX (80 objects)": "YOLOX estándar (80 objetos)",
    "Extra-Large (most accurate, slowest)": "Extragrande (más preciso, más lento)",
    "Compose higher-level actions from the spatial relationships between detected objects. Example: if object A appears inside region B a certain number of times, fire action X. Each row is one spatial condition; multiple rows with the same Event Name must ALL be satisfied together (AND logic). Window = how many seconds of frames to smooth over (reduces flicker). Persist = how long to keep an object 'alive' after YOLO loses sight of it (handles occlusion). Saved to composition_rules.yaml next to the application.": "Combina relaciones espaciales entre objetos detectados para crear acciones. Cada fila define una condición; las filas con el mismo nombre deben cumplirse juntas. Ventana suaviza detecciones durante varios segundos. Persistencia mantiene un objeto por un tiempo breve cuando YOLO deja de verlo. Las reglas se guardan en composition_rules.yaml.",
    "Filter:": "Filtro:",
    "Select All Visible": "Seleccionar todos los visibles",
    "Deselect All": "Quitar selección",
    "0 selected": "0 seleccionados",
    "Don't show this warning again": "No volver a mostrar esta advertencia",
    "Waiting for the detection stage (objects / actions)…": "Esperando la detección de objetos y acciones…",
    "⏸ Freeze": "⏸ Congelar",
    "⏭ Live": "⏭ En vivo",
    "Input Videos": "Videos de entrada",
    "Add Videos": "Agregar videos",
    "Remove Selected": "Quitar seleccionados",
    "Clear All": "Quitar todos",
    "Output base name:": "Nombre del video final:",
    "empty = <video>_highlight.mp4": "vacío = <video>_highlight.mp4",
    "name": "nombre",
    "Presets:": "Presets:",
    "Save": "Guardar",
    "Load": "Cargar",
    "Delete": "Eliminar",
    "Processing Time Range": "Rango de tiempo para analizar",
    "Set time range in percentages (0-100%) - loads actual times when video is selected": "Define el rango en porcentajes (0–100 %); los tiempos aparecen al seleccionar un video",
    "Start:": "Inicio:",
    "End": "Fin",
    "Selection: Full video": "Selección: video completo",
    "First 5min": "Primeros 5 min",
    "Last 5min": "Últimos 5 min",
    "Last 10min": "Últimos 10 min",
    "Middle": "Mitad",
    "Live detection preview (separate window)": "Vista previa de detección en vivo (otra ventana)",
    "Progress": "Progreso",
    "Download Videos from Website": "Descargar videos de una página web",
    "Page URL:": "URL de la página:",
    "Pick Videos from Page…": "Seleccionar videos de la página…",
    "Download All": "Descargar todos",
    "Download": "Descargar",
    "Process only specific time range": "Analizar solo un rango de tiempo",
    "Quick presets:": "Rangos rápidos:",
    "Full video": "Video completo",
    "Download Time Range": "Rango de tiempo para descargar",
    "Download:": "Descargar:",
    "Same range as processing": "El mismo rango del análisis",
    "Specific range (seconds)": "Rango específico (segundos)",
    "Start time (seconds):": "Inicio (segundos):",
    "End time (seconds):": "Fin (segundos):",
    "Save directory:": "Carpeta de destino:",
    "Browse...": "Examinar...",
    "After download:": "Después de descargar:",
    "Don't process": "No analizar",
    "Process each video as it downloads": "Analizar cada video al descargarlo",
    "Process all after downloads finish": "Analizar todos al terminar la descarga",
    "Concurrent downloads:": "Descargas simultáneas:",
    "Automatically add downloaded videos to file list": "Agregar videos descargados a la lista",
    "Force reprocess (ignore cache)": "Reiniciar análisis (ignorar caché)",
    "Basic Settings": "Configuración básica",
    "Scoring Points": "Puntos de selección",
    "Duration && Cutting": "Duración y cortes",
    "Max highlight duration (s):": "Duración máxima del resumen (s):",
    "Exact duration (s, 0 = off):": "Duración exacta (s, 0 = desactivada):",
    "Clip time (s, 0 = auto):": "Duración de clip (s, 0 = automático):",
    "Best moments ↔ Full story:": "Mejores momentos ↔ historia completa:",
    "Auto-Segmentation Settings": "Segmentación automática",
    "Min clip length (s):": "Duración mínima de clip (s):",
    "Max clip length (s):": "Duración máxima de clip (s):",
    "Merge gap (s):": "Separación para unir clips (s):",
    "Object detection:": "Detección de objetos:",
    "Load Labels": "Cargar etiquetas",
    "Action keywords:": "Acciones de interés:",
    "Transcript keywords:": "Palabras de interés:",
    "Only score actions when objects detected": "Puntuar acciones solo si se detectan objetos",
    "Combine highlights from all processed videos into one video": "Unir los resúmenes de todos los videos procesados",
    "Transcript && Subtitles": "Transcripción y subtítulos",
    "Transcript Settings": "Configuración de transcripción",
    "Enable transcript processing": "Activar transcripción",
    "Use transcript:": "Usar transcripción:",
    "Source language:": "Idioma original:",
    "Whisper model:": "Modelo Whisper:",
    "Subtitle Settings": "Configuración de subtítulos",
    "Generate subtitles (.srt)": "Generar subtítulos (.srt)",
    "Create subtitles:": "Crear subtítulos:",
    "Target language:": "Idioma de destino:",
    "Advanced": "Avanzado",
    "Motion Recognition": "Detección de movimiento",
    "Frame skip:": "Intervalo entre fotogramas:",
    "VR side-by-side optimization": "Optimización VR de lado a lado",
    "Object Recognition": "Detección de objetos",
    "Import model…": "Importar modelo…",
    "Community models…": "Modelos de la comunidad…",
    "Detector type:": "Tipo de detector:",
    "Detector model size:": "Tamaño del modelo:",
    "Object model:": "Modelo de objetos:",
    "Confidence threshold:": "Umbral de confianza:",
    "Action Recognition": "Detección de acciones",
    "Backend:": "Motor:",
    "Models:": "Modelos:",
    "R3D model variant:": "Variante del modelo R3D:",
    "Composition Rules": "Reglas de composición",
    "+ Add Spatial": "+ Agregar regla espacial",
    "+ Add Signal": "+ Agregar señal",
    "Save Rules": "Guardar reglas",
    "Bounding Box Visualization": "Visualización de cuadros de detección",
    "ℹ️ Enable bounding boxes, creates new file with extension _annotated.mp4 for debugging": "ℹ️ Los cuadros de detección generan un archivo _annotated.mp4 para depuración",
    "Draw bounding boxes for object detection": "Dibujar cuadros de objetos detectados",
    "Draw labels for action recognition": "Dibujar etiquetas de acciones",
    "Write a highlight report": "Generar reporte del resumen",
    "…and describe each clip": "…y describir cada clip",
    "…and tell each chapter (slow)": "…y narrar cada capítulo (lento)",
    "Served at (optional):": "Disponible en (opcional):",
    "Video reachable at (optional):": "Video accesible en (opcional):",
    "LLM Chat": "Chat con IA",
    "Train": "Entrenar",
    "Avoid": "Excluir",
    "Avoid People": "Excluir personas",
    "Enable face recognition": "Activar reconocimiento facial",
    "When found:": "Al encontrarlas:",
    "🔄 Refresh from face database": "🔄 Actualizar desde la base de rostros",
    "🔍 Scan video for faces": "🔍 Buscar rostros en el video",
    "🗑 Clear faces": "🗑 Borrar rostros",
    "About": "Acerca de",
    "Video Output": "Exportación de video",
    "Cut / encode:": "Codificador:",
    "Compute": "Procesamiento",
    "Prefer:": "Preferir:",
    "Timeline Viewer": "Abrir línea de tiempo",
    "Highlight Report": "Abrir reporte",
    "AI Summary": "Resumen con IA",
    "Simple view": "Vista simple",
    "Cancel": "Cancelar",
    "Report Only": "Solo reporte",
    "Run Highlighter": "Analizar",
    "Keep temp clips: ON": "Conservar clips temporales: sí",
    "Keep temp clips: OFF": "Conservar clips temporales: no",
    "Export clips: ON": "Exportar clips: sí",
    "Export clips: OFF": "Exportar clips: no",
    "Debug log": "Registro de depuración",
    "Log Output:": "Registro del proceso:",
    "Drop a video here": "Arrastra un video aquí",
    "or click to choose a file  ·  mp4, mov, mkv, avi": "o haz clic para seleccionar un archivo · mp4, mov, mkv, avi",
    "Find and explain the moments that matter": "Encuentra y explica los mejores momentos",
    "Footage stays on your disk. Analyze finds strong moments with built-in defaults (motion peaks and loudness), writes a highlight reel plus separate clips, and shows why each moment scored — timeline, report, and chat. Open Detailed settings only when you need full control or to teach it what to look for.": "Tus videos permanecen en esta computadora. Analizar detecta los mejores momentos con movimiento y audio, crea un resumen y clips separados, y explica cada selección en la línea de tiempo, el reporte y el chat. Abre la configuración detallada cuando quieras ajustar los criterios de análisis.",
    "Progress shows up here while Analyze runs.": "El progreso aparecerá aquí durante el análisis.",
    "Highlight length": "Duración del resumen",
    "Short  — about 1–2 minutes": "Corto · 1–2 minutos",
    "Medium  — about 4 minutes": "Medio · 4 minutos",
    "Longer  — about 7 minutes": "Largo · 7 minutos",
    "Remove": "Quitar",
    "Ready": "Listo",
    "Analyze": "Analizar",
    "Open timeline": "Abrir línea de tiempo",
    "Open report": "Abrir reporte",
    "Ask about this video": "Preguntar sobre este video",
    "Detailed settings…": "Configuración detallada…",
    "LLM Settings": "Configuración de IA",
    "Backend:": "Proveedor:",
    "Model:": "Modelo:",
    "Refresh": "Actualizar",
    "Connect": "Probar conexión",
    "Ollama host:": "Servidor Ollama:",
    "GGUF path:": "Ruta de GGUF:",
    "mmproj path:": "Ruta de mmproj:",
    "No video context": "Sin datos del video",
    "Enable reasoning": "Activar razonamiento",
    "Stats": "Estadísticas",
    "Load Cache": "Cargar caché",
    "Show Context": "Ver datos enviados",
    "Visual Search": "Búsqueda visual",
    "Search for:": "Buscar:",
    "From:": "Desde:",
    "Clear": "Limpiar",
    "Free chat": "Chat libre",
    "Download and install": "Descargar e instalar",
    "Get it": "Obtener actualización",
    "Skip this version": "Omitir esta versión",
    "Try Pro free": "Probar Pro gratis",
    "Don't suggest Pro": "No sugerir Pro",
    "Updates": "Actualizaciones",
    "Check for a newer version automatically": "Buscar actualizaciones automáticamente",
    "Check now": "Buscar ahora",
    "Contact & Support": "Contacto y soporte",
    "Need help, found a bug, or have a feature request? Reach us here:": "¿Necesitas ayuda, encontraste un error o tienes una sugerencia? Contáctanos aquí:",
    "Legal": "Información legal",
    "edit them in Advanced → Composition Rules": "edítalas en Avanzado → Reglas de composición",
    "Complete!": "¡Completado!",
    "Output saved to:": "Video guardado en:",
}


def _translate(value: str, language: str) -> str:
    if language != "es":
        return value
    if value in ES:
        return ES[value]
    if value.startswith("Loaded: "):
        return "Cargado: " + value[8:]
    match = re.fullmatch(r"(\d+) videos ready — press Analyze", value)
    if match:
        return f"{match.group(1)} videos listos · haz clic en Analizar"
    if value == "1 video ready — press Analyze":
        return "1 video listo · haz clic en Analizar"
    return value


def translate_tree(root: QWidget, language: str) -> None:
    """Translate cataloged widget text; preserve source text for switching back."""
    for widget in [root, *root.findChildren(QWidget)]:
        if isinstance(widget, QTabWidget):
            for i in range(widget.count()):
                page = widget.widget(i)
                source = page.property("i18n_tab_source")
                if source is None:
                    source = widget.tabText(i)
                    page.setProperty("i18n_tab_source", source)
                widget.setTabText(i, _translate(source, language))
        if isinstance(widget, QComboBox):
            for i in range(widget.count()):
                source = widget.itemData(i, 1000)
                if source is None:
                    source = widget.itemText(i)
                    widget.setItemData(i, source, 1000)
                widget.setItemText(i, _translate(source, language))
        if isinstance(widget, (QLabel, QAbstractButton, QGroupBox)):
            source = widget.property("i18n_source")
            current = widget.text() if not isinstance(widget, QGroupBox) else widget.title()
            if source is None or current != widget.property("i18n_last"):
                source = current
                widget.setProperty("i18n_source", source)
            translated = _translate(source, language)
            if isinstance(widget, QGroupBox):
                widget.setTitle(translated)
            else:
                widget.setText(translated)
            widget.setProperty("i18n_last", translated)
        if isinstance(widget, (QLineEdit, QTextEdit)):
            source = widget.property("i18n_placeholder_source")
            current = widget.placeholderText()
            if source is None or current != widget.property("i18n_placeholder_last"):
                source = current
                widget.setProperty("i18n_placeholder_source", source)
            translated = _translate(source, language)
            widget.setPlaceholderText(translated)
            widget.setProperty("i18n_placeholder_last", translated)
