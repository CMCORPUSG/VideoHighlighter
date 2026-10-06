# Validación de VideoHighlighter para gaming largo

## Flujo de uso

1. Inicia `python main.py`. En la vista simple, carga un VOD y selecciona **Gaming largo**.
2. Elige una duración aproximada para el resumen. Por ejemplo, **15 min** es el objetivo del conjunto de eventos, no la duración de cada clip. Un evento puede superar el presupuesto si así conserva su desenlace.
3. Elige **Ninguno** para procesamiento local, **Gemini API** para revisión semántica opcional u **Ollama local** para el chat. Configura Gemini si lo seleccionas.
4. Pulsa **Analizar**. Al terminar, abre el video final, la carpeta de clips o **Revisar y calificar clips**.
5. Marca cada clip como **Excelente**, **Bueno** o **Malo**. Registra los eventos importantes que faltaron con su segundo aproximado. La ventana muestra precisión percibida, falsos positivos, duración media, llamadas Gemini y uso de caché.

## Selección de eventos

El análisis existente produce señales de movimiento, audio, cambios de escena, objetos, acciones y transcripción, cuando cada módulo está disponible. Los intervalos con evidencia temporal se agrupan antes de escoger clips. Una pausa corta une actividad próxima; una pausa mayor solo se une si comparte una etiqueta específica. Los cambios de escena aportan puntuación pero por sí solos no delimitan un evento. El contexto inicial orientativo es de 12–18 segundos y el final de 15–20 segundos; la actividad relacionada puede extenderlos. No existe una duración fija por clip.

Cada candidato recibe una puntuación local que combina contraste con el fondo del VOD, varias señales, actividad a lo largo del intervalo y continuidad hacia el final. Se penalizan intervalos muy cortos y actividad aislada de movimiento. Se ordenan los candidatos, se filtran los de evidencia más débil y se limita la repetición de etiquetas específicas. El presupuesto de duración elige eventos completos: nunca recorta un evento para alcanzar una cifra exacta.

Estas señales son aproximaciones, no una prueba de que ocurrió un momento interesante. Un combate silencioso puede ser valioso y una cámara en movimiento puede no serlo. Por eso la selección conserva la revisión humana y el modo semántico opcional.

## Gemini opcional

En **Configurar Gemini** se puede guardar, cambiar, mostrar temporalmente o eliminar la API key. La interfaz oculta el campo por defecto y usa `keyring` / Administrador de credenciales de Windows. `GEMINI_API_KEY` sigue disponible como alternativa de entorno. La clave no entra en `config.yaml`, Git, reportes ni caché. **Probar conexión** hace una solicitud breve de texto.

Los modelos iniciales son `gemini-3.5-flash-lite` y `gemini-3.5-flash`; el campo permite otro modelo compatible. **Desactivado** evita consultas. **Solo cuando sea necesario** prioriza candidatos de puntuación incierta entre los mejores candidatos locales. **Candidatos finales** revisa los mejores por puntuación. Los límites predeterminados son 30 candidatos y 30 llamadas por video, configurables entre 0 y 100. Las llamadas se cuentan y persisten antes de enviar la solicitud, incluso si falla. Al llegar al máximo, el resto se decide localmente.

Se envían solo datos compactos de cada candidato y hasta cinco fotogramas JPEG pequeños (320 px de ancho como máximo), más transcripción y detecciones acotadas si existen. No se envía el VOD completo ni se usa Gemini para renderizar. Se solicita JSON estructurado y se validan tipos y valores de relevancia, categoría, importancia, interés, consecuencia, integridad del evento y motivo. Un error de red o respuesta inválida deja el ranking local operativo. La puntuación final combina 65 % local y 35 % semántica, con penalizaciones para candidatos irrelevantes o incompletos.

Las respuestas se guardan por identidad del archivo de video, modelo e intervalo del candidato en `cache/gemini_events/`. La misma revisión puede reutilizarse sin otra llamada. El contador de llamadas es persistente para ese video; eliminar la caché reinicia ese contador y sus respuestas.

## Feedback y criterio de aceptación

Las calificaciones se guardan solo en `cache/clip_feedback/`, con video, intervalo, duración, puntuaciones, señales y evaluación Gemini cuando exista. No se entrena ningún modelo automáticamente. **Precisión percibida** = (clips excelentes + buenos) / clips evaluados; los malos son falsos positivos declarados por el usuario. Los eventos omitidos se registran aparte.

Para validar calidad en el VOD real de 2 h 10 min, revisa todos los clips exportados y anota los momentos importantes que faltaron. La meta inicial es al menos 70 % de clips relevantes; la deseada, 80 %. Estas cifras solo pueden afirmarse después de calificar el material. La caché existente permite volver a probar el ranking sin repetir el análisis completo. La selección local carece de una comprensión visual fiable de objetivos, riesgo y resultado cuando no hay detectores útiles o revisión semántica; en ese caso se deben interpretar las métricas y corregir con el feedback.

## Comprobaciones técnicas

Ejecuta las pruebas dirigidas con `python -m pytest -q tests/test_event_quality.py tests/test_gaming_workflow.py tests/test_simple_start.py tests/test_highlight_swap.py`. Comprueba en una ejecución real que CUDA/NVIDIA y NVENC siguen activos, los clips y el resumen se exportan, la caché se reutiliza, la línea de tiempo y el reporte abren, y el modo **Ninguno** funciona sin conexión. Usa el registro para distinguir análisis y exportación. El sistema conserva los modos anteriores de ventanas fijas y segmentación automática para proyectos previos.
