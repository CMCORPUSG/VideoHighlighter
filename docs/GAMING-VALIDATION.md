# Validación de videos gaming largos

## Configuración

1. Abre `python main.py` desde la rama `feat/video-highlighter-validation`.
2. En la vista simple, selecciona **Gaming largo** y una duración de 5, 10, 15, 20 minutos o personalizada.
   La duración es un objetivo aproximado para el resumen, no una longitud fija por clip.
   En la vista detallada, **Tipo de segmento: Eventos completos** es el valor predeterminado.
   Los modos anteriores de ventanas fijas y segmentación automática siguen disponibles para proyectos previos.
3. En la configuración avanzada, deja **Codificador: Automático** para intentar NVENC con fallback a CPU. La opción **CPU** conserva la ruta compatible con material VR.
4. Si quieres contexto de Gemini, define `GEMINI_API_KEY` en tu entorno de Windows, elige **Gemini API** y haz clic en **Probar conexión**. Sin clave, el análisis local funciona normalmente. La clave no se guarda en `config.yaml`.

## Matriz de pruebas reales

| Prueba | Entrada | Qué comprobar |
| --- | --- | --- |
| P1 | Gaming 10–20 min, 1080p | Carga, señales visuales y audio, clips, reporte, MP4, sincronización |
| P2 | Gaming 60 min, 1080p/1440p | Memoria estable, duración objetivo, CUDA, NVENC y fallback |
| P3 | Gaming 2–3 h, 1080p/1440p | Cancelar después de una etapa, reiniciar Windows, continuar y comparar salida |

Conserva para cada ejecución: tiempo total y de exportación, modelo GPU, VRAM, CPU y RAM, segmentos elegidos, duración final, resolución, FPS, codificador registrado, y errores del registro. Comprueba visualmente una pelea intensa sin voz: debe poder entrar por señales de movimiento, escenas, objetos o audio, aunque no haya transcripción.

## Revisión de eventos completos

Revisa cada clip por separado. Debe mostrar contexto anterior, inicio, desarrollo,
momento principal y consecuencia. Verifica que una pelea y su pantalla de resultado
no se corten por un cambio de escena, que dos eventos separados por una pausa real
no se dupliquen y que el último clip no termine abruptamente por alcanzar el
objetivo de duración. Si las señales locales no distinguen el comienzo o el final
de una secuencia, ajusta los límites en la línea de tiempo antes de publicar.

La agrupación usa señales locales y etiquetas detectadas; no puede certificar por
sí sola que el relato de cada clip sea comprensible. Gemini permanece opcional
para revisar el contexto mediante los resultados procesados, sin enviar el VOD.

El caché terminado se reutiliza antes de volver a exportar. La transcripción y el análisis de escenas/movimiento también se guardan por etapa; una etapa interrumpida se repite para evitar resultados incompletos. La identidad del video y la firma de los parámetros se comprueban antes de reutilizar cada etapa.
