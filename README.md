# Accident Detection System — Equipo 8
**Samsung Innovation Campus | Grupo 8 | SIC 2025**

Sistema inteligente de predicción de riesgo de accidente vial en tiempo real. Procesa video de conducción frame a frame usando un pipeline de 5 etapas: preprocesamiento, percepción espacial, modelado espaciotemporal, predicción con incertidumbre y un agente adaptativo de inferencia.

---

## Resultados

### HUD del Agente — Niveles de Riesgo

| Riesgo Bajo | Riesgo Medio | Riesgo Alto |
|:-----------:|:------------:|:-----------:|
| ![overlay_low](LINK_DRIVE_overlay_test_low.jpg) | ![overlay_medium](LINK_DRIVE_overlay_test_medium.jpg) | ![overlay_high](LINK_DRIVE_overlay_test_high.jpg) |

### Curvas de Entrenamiento

| ResNet18+SPP | SegFormer-B0 | YOLO Evaluation |
|:------------:|:------------:|:---------------:|
| ![resnet](LINK_DRIVE_resnet_training_curves.png) | ![segformer](LINK_DRIVE_segformer_training_curves.png) | ![yolo](LINK_DRIVE_yolo_evaluation.png) |

### Videos del Pipeline Completo

| Escena | Video completo (HUD + alertas) | Segmentación semántica |
|--------|-------------------------------|------------------------|
| Día urbano | [dia1_full.mp4](LINK_DRIVE_dia1_full.mp4) | [dia1_seg.mp4](LINK_DRIVE_dia1_seg.mp4) |
| Peatones | [DiaPeatones_full.mp4](LINK_DRIVE_DiaPeatones_full.mp4) | [DiaPeatones_seg.mp4](LINK_DRIVE_DiaPeatones_seg.mp4) |
| Lluvia mediodía | [MedioDiaLluvia_full.mp4](LINK_DRIVE_MedioDiaLluvia_full.mp4) | [MedioDiaLluvia_seg.mp4](LINK_DRIVE_MedioDiaLluvia_seg.mp4) |
| Prueba agente | [video_output.mp4](LINK_DRIVE_video_output.mp4) | — |

---

## Arquitectura del Pipeline

```
Video BDDA
    │
    ▼
01_preprocessing      → CLAHE condicional · Resize 416×416 · Normalización ImageNet
    │
    ▼
02_spatial_perception → YOLOv8 (detección) · SegFormer-B0 (segmentación) · ResNet18+SPP (features)
    │
    ▼
03_spatiotemporal     → GCNEncoder (grafo de objetos) · RiskLSTM (secuencia temporal)
    │
    ▼
04_prediction_alert   → BNNRiskPredictor (MC Dropout · incertidumbre)
    │
    ▼
05_agent              → Strategy pattern adaptativo · HUD overlay · AlertLogger
    │
    ▼
video_output.mp4 + alerts.csv
```

---

## Métricas por Módulo

| Módulo | Modelo | Métrica | Valor |
|--------|--------|---------|-------|
| Percepción espacial | SegFormer-B0 | mIoU | 0.4661 |
| Percepción espacial | ResNet18+SPP | val_loss | 0.3661 |
| Percepción espacial | YOLOv8n | mAP@0.5 | 0.0886 |
| Dataset curado | BDD100K | imágenes 416×416 | 61,345 |

---

## Estructura del Repositorio

```
Equipo8-Grupo8-SIC-2025-/
├── notebooks/
│   ├── S1_data_engineering.ipynb       # Ingeniería de datos BDD100K
│   ├── 01_preprocessing.ipynb          # Pipeline CLAHE + curación
│   ├── 02_spatial_perception.ipynb     # YOLO · SegFormer · ResNet18
│   ├── 03_spatiotemporal_modeling.ipynb# GCNEncoder + RiskLSTM
│   ├── 04_prediction_alert.ipynb       # BNN con MC Dropout
│   ├── 05_agent.ipynb                  # Agente de inferencia en video
│   ├── convert_to_yolo_baseline.py     # Conversión de etiquetas BDD→YOLO
│   ├── RESUMEN_PIPELINE.md             # Documentación técnica completa
│   └── src/
│       ├── model_manager.py            # Lazy loading + offloading VRAM
│       ├── perception.py               # Motor de percepción
│       ├── risk_estimator.py           # Estimación de riesgo
│       ├── overlay.py                  # Renderizado HUD
│       ├── alert_logger.py             # Logging CSV + JSON
│       └── agent_state.py             # Máquina de estados (histéresis)
├── requirements.txt
└── .gitignore
```

---

## Instalación

```bash
git clone https://github.com/Rogelio756/Equipo8-Grupo8-SIC-2025-.git
cd Equipo8-Grupo8-SIC-2025-
git checkout estrategia-1

python -m venv env_samsung
source env_samsung/bin/activate   # Linux/macOS
# .\env_samsung\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## Estado del Proyecto

| Etapa | Notebook | Estado |
|-------|----------|--------|
| Data Engineering | S1_data_engineering.ipynb | Completado |
| Preprocesamiento CLAHE | 01_preprocessing.ipynb | Completado |
| Percepción Espacial | 02_spatial_perception.ipynb | Completado |
| Modelado Espaciotemporal | 03_spatiotemporal_modeling.ipynb | Completado |
| Predicción con Incertidumbre | 04_prediction_alert.ipynb | Completado |
| Agente de Inferencia | 05_agent.ipynb | Completado |

---

## Equipo

**Equipo 8 — Grupo 8 | Samsung Innovation Campus 2025**
