# Equipo8-Grupo8-SIC-2025 — Pipeline de Predicción de Accidentes

## Arquitectura general

El sistema procesa video de conducción del dataset BDDA (BDD Driver Attention)
para estimar en tiempo real el riesgo de accidente fotograma a fotograma.
El pipeline se compone de 5 cuadernos de entrenamiento y un agente de inferencia:

```
01_preprocessing      → Curación de imágenes BDD100K (CLAHE, resize 416×416)
02_spatial_perception → Entrenamiento: YOLOv8 · SegFormer-B0 · ResNet18+SPP
03_spatiotemporal     → Entrenamiento: GCNEncoder + RiskLSTM sobre grafos de objetos
04_prediction_alert   → Entrenamiento: BNNRiskPredictor (MC Dropout, incertidumbre)
05_agent              → Agente de inferencia sobre video BDDA en tiempo real
```

El agente implementa un **strategy pattern** adaptativo: según el nivel de riesgo
actual activa más o menos modelos, balanceando precisión y latencia bajo la
restricción de 6 GB VRAM (GTX 1660 Ti).

```
Video BDDA
    │
    ▼
PerceptionEngine ──── YOLO (detección)
    │            ──── SegFormer (segmentación semántica)   [medium/high]
    │            ──── ResNet18+SPP (features de escena)    [high]
    │
    ▼
RiskEstimator ─────── GCNEncoder  (grafo de objetos → embedding)
    │          ─────── RiskLSTM   (secuencia temporal → score)
    │          ─────── BNNRiskPredictor (incertidumbre MC Dropout)
    │
    ▼
AgentState ─────────── Histéresis (THRESH_MEDIUM=0.62, THRESH_HIGH=0.68)
    │
    ▼
OverlayRenderer ─────── Máscara seg · Boxes YOLO · HUD (riesgo/nivel/sigma)
    │
    ▼
AlertLogger ────────── CSV de alertas + JSON de métricas
    │
    ▼
video_output.mp4
```

---

## Módulos src/

### `model_manager.py`
**Propósito:** Registro centralizado de modelos con lazy loading y offloading de VRAM.

**Clase principal:** `ModelRegistry`

| Método | Descripción |
|--------|-------------|
| `register_model(name, path, loader_fn, vram_gb)` | Registra un modelo sin cargarlo. Verifica existencia del checkpoint. |
| `load_model(name)` | Carga en GPU (desde disco, o mueve de CPU si ya está en RAM). |
| `offload_model(name)` | Mueve GPU → CPU, libera VRAM. El modelo permanece en RAM para recarga rápida. |
| `offload_all()` | Offloadea todos los modelos en GPU. |
| `status()` | Tabla con estado (cargado/GPU) y VRAM estimada de cada modelo. |
| `_print_vram(contexto)` | Imprime VRAM real PyTorch + libre + total. |

**Input → Output:** `checkpoint_path + loader_fn` → `nn.Module` en GPU lista para inferencia.

**Costes VRAM estimados:**

| Modelo | VRAM |
|--------|------|
| yolo | 0.6 GB |
| segformer | 1.2 GB |
| resnet | 0.3 GB |
| gcn | 0.1 GB |
| lstm | 0.1 GB |
| bnn | 0.2 GB |
| **Total high** | **2.5 GB** |

---

### `agent_state.py`
**Propósito:** Estado del agente — mantiene nivel de riesgo con histéresis y decide qué modelos activar.

**Clase principal:** `AgentState`

| Método | Descripción |
|--------|-------------|
| `update(frame, risk_score)` | Actualiza buffer, historial y nivel. Retorna nivel actual. |
| `_update_level(score)` | Subida inmediata al superar umbral; bajada tras `HYSTERESIS_DOWN=5` frames consecutivos bajo umbral. |
| `active_models()` | Strategy pattern — lista de modelos a activar según nivel. |
| `status()` | Imprime nivel, frames procesados, último score y promedio. |

**Umbrales (calibrados sobre video BDDA):**
- `THRESH_MEDIUM = 0.62` (p50 del rango real BNN)
- `THRESH_HIGH   = 0.68` (p80 del rango real BNN)
- `HYSTERESIS_DOWN = 5` frames para bajar un nivel

**Input → Output:** `(frame, risk_score: float)` → `level: str` ∈ `{"low", "medium", "high"}`

---

### `perception.py`
**Propósito:** Motor de percepción — ejecuta los modelos visuales según `active_models`.

**Clase principal:** `PerceptionEngine`

| Método | Descripción |
|--------|-------------|
| `__init__(registry)` | Inicializa procesador SegFormer y constantes de normalización. |
| `_apply_clahe(frame, mode)` | CLAHE condicional: `night` (clip=3.0) · `weather` (clip=2.5) · `normal` (sin cambio). |
| `_preprocess_seg(frame)` | Resize 320×320 · RGB · normalización ImageNet · tensor CPU. |
| `_preprocess_res(frame)` | Resize 384×288 · RGB · normalización ImageNet · tensor CPU. |
| `run(frame, active_models, clahe_mode)` | Pipeline completo: CLAHE → YOLO → SegFormer → ResNet → resultado. |

**Reconstrucción automática desde state_dict:**
- `SegFormer`: reconstruye `SegformerForSemanticSegmentation` (B0-cityscapes, 19 clases) en primer uso.
- `ResNet`: reconstruye `ResNet18SPP` (pool_sizes=[1,2,4]) en primer uso.
- Ambos se guardan en el registry para no reconstruir en frames posteriores.

**Input → Output:** `(frame_bgr, active_models, clahe_mode)` →
```python
{
  "frame_clahe": np.ndarray (BGR),
  "boxes":       ultralytics.Results  (detecciones YOLO),
  "mask":        torch.Tensor HxW int64 (clases Cityscapes 0-18),
  "features":    torch.Tensor (1, 2048, H/32, W/32),
  "risk_score":  None  (calculado por RiskEstimator),
  "clahe_mode":  str
}
```

---

### `risk_estimator.py`
**Propósito:** Estimación de riesgo con pipeline GCN → LSTM → BNN sobre los objetos YOLO.

**Clases internas:** `GCNEncoder` · `RiskLSTM` · `MCDropout` · `BNNRiskPredictor`

**Clase principal:** `RiskEstimator`

| Método | Descripción |
|--------|-------------|
| `__init__(registry, seq_len=10)` | Inicializa buffer circular de embeddings. |
| `_boxes_to_graph(boxes_result, frame_shape)` | Convierte detecciones YOLO en grafo PyG. Nodos: 2063-dim (xyxy + one-hot + conf + mask_zone + ResNet zeros). Edges locales (<150px) y globales (<400px). |
| `_ensure_gcn/lstm/bnn(model)` | Reconstruye arquitectura desde state_dict si es `OrderedDict`, la mueve a GPU y actualiza el registry. |
| `estimate(boxes_result, frame_shape)` | Corre GCN→embedding→buffer. Si buffer < seq_len retorna (0.5, 0.0). Si completo corre LSTM+BNN. |

**Dimensiones:**
- `GCNEncoder`: `in_dim=2063 → hidden=256 → out_dim=128`
- `RiskLSTM`: `input=128 → hidden=256 (2 capas) → (B,1)`
- `BNNRiskPredictor`: `input=128 → hidden=64 → scalar` + MC Dropout (20 muestras)

**Input → Output:** `(boxes_result, frame_shape)` → `(risk_score: float, sigma: float)`

---

### `overlay.py`
**Propósito:** Renderizado visual — superpone máscara semántica, bounding boxes y panel HUD.

**Clase principal:** `OverlayRenderer`

| Método | Descripción |
|--------|-------------|
| `_draw_seg_mask(frame, mask)` | Colorea la máscara con la paleta Cityscapes (indexación vectorizada) y la mezcla con `alpha_seg=0.45`. |
| `_draw_boxes(frame, boxes)` | Dibuja bounding boxes con etiquetas de clase COCO (nombres propios de YOLO, no Cityscapes). Shadow negro + texto blanco. |
| `_draw_hud(frame, risk_score, level, frame_count, uncertainty)` | Panel semitransparente con `Risk`, `Level`, `Frame`, y `sigma` (si disponible). Color según nivel. |
| `render(frame, perception_out, level, frame_count, uncertainty)` | Pipeline completo: máscara → boxes → HUD. |

**Colores HUD por nivel:**
- `low` → verde `(0, 200, 0)`
- `medium` → naranja `(0, 165, 255)`
- `high` → rojo `(0, 0, 220)`

**Input → Output:** `(frame_bgr, perception_out, level, frame_count)` → `frame_anotado: np.ndarray BGR`

---

### `alert_logger.py`
**Propósito:** Observer del agente — registra alertas con debouncing y persiste métricas.

**Clase principal:** `AlertLogger`

| Método | Descripción |
|--------|-------------|
| `__init__(csv_path, json_path, debounce_frames=10)` | Inicializa contadores y crea CSV con cabecera. |
| `update(frame_count, level, risk_score, uncertainty, timestamp_ms)` | Emite alerta si `level` ∈ {medium, high} y han pasado ≥ `debounce_frames` desde la última. Escribe fila en CSV. |
| `save_metrics(video_path, fps_real)` | Escribe `video_metrics.json` con resumen completo de la sesión. |
| `summary()` | Imprime en terminal: frames por nivel, alertas emitidas y latencia media. |

**Input → Output:** `(frame_count, level, risk_score)` → alerta en CSV si pasa debouncing · `Optional[str]`

---

## Notebook 05_agent.ipynb

### Celdas

| Celda | Qué hace |
|-------|----------|
| **0 — Título** | Markdown descriptivo. |
| **1 — Imports** | Importa todas las librerías y módulos de `src/`. Configura rutas `BASE`, `CKPT`, `OUT`. |
| **2 — Selección video** | Busca todos los `.mp4` en `BDDA/training/camera_videos/` excluyendo el corrupto `922.mp4`. Selecciona el primero. |
| **3 — ModelRegistry** | Define `loader_yolo` y `loader_torch`. Registra los 6 modelos con rutas y costes VRAM. |
| **4 — Componentes** | Instancia `AgentState`, `PerceptionEngine`, `OverlayRenderer`, `AlertLogger`, `RiskEstimator`. |
| **5 — Inspección video** | Lee metadatos del video: resolución, FPS fuente, frames totales, duración. |
| **6 — Loop principal** | Lee frames → percepción → estimación de riesgo (GCN+LSTM+BNN) → actualiza estado → renderiza overlay → escribe video → registra alerta. Muestra progreso con `tqdm`. |
| **7 — Outputs** | Llama a `save_metrics()` y `summary()`. Lista archivos generados con tamaño. |
| **8 — Offload VRAM** | Descarga todos los modelos de GPU y muestra estado final del registry. |

### Cómo ejecutarlo

```bash
cd notebooks/
jupyter notebook 05_agent.ipynb
# Ejecutar celdas en orden: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
```

**Parámetros configurables en Celda 6:**

| Variable | Valor por defecto | Descripción |
|----------|------------------|-------------|
| `CLAHE_MODE` | `"normal"` | Modo de mejora: `"normal"`, `"night"`, `"weather"` |
| `MAX_FRAMES` | `None` | `None` = video completo · entero = limitar frames |
| `FRAME_SKIP` | `1` | Procesar 1 de cada N frames |

---

## Métricas del sistema

### FPS y modelos activos por nivel

| Nivel | Modelos activos | VRAM total | FPS estimado |
|-------|----------------|------------|--------------|
| `low` | yolo | ~0.6 GB | ~25–30 fps |
| `medium` | yolo · segformer | ~1.8 GB | ~8–12 fps |
| `high` | yolo · segformer · resnet | ~2.1 GB | ~5–8 fps |

> `RiskEstimator` (GCN+LSTM+BNN) corre en **todos los frames** independientemente
> del nivel, con coste adicional ~0.4 GB VRAM y ~5 ms/frame.

### Umbrales de transición (calibrados en video BDDA)

| Transición | Umbral | Comportamiento |
|------------|--------|----------------|
| low → medium | `0.62` | Inmediata al superar |
| medium → high | `0.68` | Inmediata al superar |
| high → medium | `0.62` | Tras 5 frames consecutivos bajo umbral |
| medium → low | `0.62` | Tras 5 frames consecutivos bajo umbral |

### Latencia media observada (GTX 1660 Ti, 6 GB VRAM)

| Componente | Latencia aprox. |
|------------|----------------|
| YOLO (imgsz=416) | ~15 ms |
| SegFormer (320×320) | ~25 ms |
| ResNet18+SPP (384×288) | ~8 ms |
| GCN + LSTM + BNN | ~5 ms |
| Overlay + escritura | ~3 ms |
| **Total nivel low** | **~20 ms (~50 fps)** |
| **Total nivel medium** | **~45 ms (~22 fps)** |
| **Total nivel high** | **~55 ms (~18 fps)** |

> Scores BNN observados en video BDDA: rango `0.64–0.69`, sigma `0.01–0.025`.

---

## Outputs generados

Todos los archivos se guardan en:
```
notebooks/data/spatial_outputs/video_outputs/
```

### `video_output.mp4`
Video anotado con:
- Máscara semántica Cityscapes semitransparente (nivel medium/high)
- Bounding boxes YOLO con etiquetas de clase COCO y confianza
- Panel HUD con: `Risk` (score BNN), `Level` (low/medium/high), `Frame`, `sigma` (incertidumbre BNN)
- Color del HUD: verde (low), naranja (medium), rojo (high)

### `video_alerts.csv`
Registro de alertas emitidas con debouncing de 10 frames:
```
timestamp,frame,level,risk_score,uncertainty
2026-04-05T14:50:09.931,20,medium,0.6500,0.0180
2026-04-05T14:50:09.932,35,high,0.6850,0.0230
```

### `video_metrics.json`
Resumen completo de la sesión:
```json
{
  "video": "ruta/al/video.mp4",
  "fecha": "2026-04-05T14:50:09",
  "total_frames": 300,
  "fps_real": 12.5,
  "frames_por_nivel": { "low": 150, "medium": 100, "high": 50 },
  "alertas": { "medium": 10, "high": 5 },
  "latencia_ms": { "media": 45.2, "max": 120.0, "min": 18.5 }
}
```

---

## Cómo ejecutar

### Requisitos previos

```bash
# Instalar dependencias
pip install torch torchvision ultralytics transformers
pip install torch-geometric opencv-python tqdm

# Verificar GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Verificar módulos individualmente

```bash
cd notebooks/

# Verificar cada módulo con su bloque __main__
python src/model_manager.py   # Prueba load/offload
python src/agent_state.py     # Prueba transiciones de nivel
python src/perception.py      # Prueba YOLO + SegFormer + ResNet sobre imagen
python src/overlay.py         # Genera overlay_test_{low,medium,high}.jpg
python src/alert_logger.py    # Simula 50 frames y genera CSV/JSON de prueba
python src/risk_estimator.py  # Prueba GCN+LSTM+BNN sobre 15 frames simulados
```

### Ejecutar el agente completo

```bash
cd notebooks/
jupyter notebook 05_agent.ipynb
# Ejecutar todas las celdas en orden (Kernel → Restart & Run All)
```

### Ejecutar en modo rápido (primeros 100 frames)

En Celda 6 del notebook cambiar:
```python
MAX_FRAMES = 100   # en lugar de None
FRAME_SKIP = 2     # procesar 1 de cada 2 frames
```

### Rutas de checkpoints

```
notebooks/
├── yolov8n.pt                                          # YOLOv8n
└── data/spatial_outputs/
    ├── segformer_best.pt                               # SegFormer-B0 fine-tuned
    ├── resnet_best.pt                                  # ResNet18+SPP
    ├── checkpoints_03/
    │   ├── gcn_best.pt                                 # GCNEncoder
    │   └── lstm_best.pt                                # RiskLSTM
    └── prediction_outputs/checkpoints_bnn/
        └── bnn_best.pt                                 # BNNRiskPredictor
```

### Videos de entrada (BDDA)

```
notebooks/data/videos/BDDA/
├── training/camera_videos/    # 925 clips válidos (excluir 922.mp4)
├── validation/camera_videos/  # 203 clips
└── test/camera_videos/        # 306 clips
```

Resolución: 1280×720 · FPS: 30 Hz · Duración media: 10.6 s/clip · Total: ~254 min
