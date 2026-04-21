# 🚗💡 Comparación de Pipelines de Visión Computacional para la Detección de Factores de Riesgo en Accidentes Automovilísticos
### Equipo 8 · Grupo 8 · Samsung Innovation Campus 2025–2026

> **Pipeline híbrido de visión por computadora y aprendizaje profundo para la predicción de riesgo vehicular en tiempo real a partir de footage de dashcam, validado sobre el dataset BDD100K (61,345 imágenes) y 925 clips reales del conjunto BDDA.**

---

## 👋 Bienvenida

¡Bienvenidos al repositorio oficial del **Equipo 8** para el **Samsung Innovation Campus 2025–2026**!

Este proyecto aborda una problemática de alto impacto social: en 2024 se registraron **374,949 accidentes de tránsito en México**, dejando 85,980 heridos y 4,656 fallecidos (INEGI, 2024). Los sistemas convencionales operan de forma **reactiva**, sin memoria temporal ni comprensión profunda del contexto vial.

Nuestra respuesta es un sistema inteligente basado en visión computacional que analiza escenas de conducción en tiempo real y genera un **Risk Score preventivo** antes de que ocurra un evento. Tres integrantes desarrollaron estrategias de pipeline independientes sobre el mismo dataset, convirtiendo el proyecto en un **estudio comparativo de tres enfoques**.

---

## 📄 Documentación del Proyecto

📑 **[Leer el reporte de investigación completo (Google Drive)](https://drive.google.com/file/d/1SljURoOS64QAmCp11E4AWskLvXaybuZ8/view?usp=sharing)**

El documento cubre: introducción y problemática, estado del arte, metodología de las tres estrategias, resultados experimentales, discusión comparativa y trabajo futuro — en formato de reporte académico Samsung Innovation Campus 2025–2026.

---

## 🧠 Resumen Técnico

La hipótesis central del proyecto establece que, al clasificar con precisión los factores de riesgo del entorno y las trayectorias vehiculares en tiempo real, es posible transformar la prevención de siniestros de una suposición estadística a una **detección determinista de riesgos**.

### Pipeline de 5 Etapas (Estrategia I)

| Etapa | Componente | Descripción |
|-------|-----------|-------------|
| **01 · Preprocesamiento** | CLAHE Condicional | 61,345 imágenes BDD100K → 416×416, split 70/15/15. CLAHE en diurnas, normalización en nocturnas |
| **02 · Percepción Espacial** | YOLOv8n + SegFormer-B0 + ResNet-18+SPP | Detección de objetos (nodos GCN), segmentación semántica (val mIoU=0.47) y extracción de features |
| **03 · Modelado Espaciotemporal** | GCNEncoder + RiskLSTM | Grafo de interacciones vehiculares + secuencias de 10 frames → **Pearson r=0.916**, MAE=0.145 |
| **04 · Predicción con Incertidumbre** | BNN con MC Dropout (T=20) | Risk Score continuo + intervalo de confianza → Pearson r=0.615 en test |
| **05 · Agente de Percepción-Acción** | FSM + Strategy Pattern | Decisiones en tiempo real: ~44 FPS (LOW) · ~12 FPS (MEDIUM) · ~8.1 FPS (HIGH) |

**Dataset:** [BDD100K](https://www.vis.xyz/bdd100k/) — 61,345 imágenes curadas  
**Validación OOD:** BDDA Xia2018 — 925 clips dashcam reales, 1280×720 px  
**Hardware:** NVIDIA GTX 1660 Ti (6GB VRAM) · CUDA 12.1

---

## 🌿 Estructura del Repositorio — Explora las Ramas

El repositorio está organizado en **ramas independientes por estrategia**. Cada una contiene los notebooks, resultados y documentación específicos de ese pipeline:

main  ←  Estás aquí · Visión general y comparativa del proyecto
│
├── 📂 strategy-roy / strategy-I
│     Pipeline completo de 5 etapas (Rogelio)
│     ├── 01_preprocessing.ipynb          — CLAHE condicional
│     ├── 02_spatial_perception.ipynb     — YOLOv8n + SegFormer + ResNet
│     ├── 03_spatiotemporal_modeling.ipynb — GCNEncoder + RiskLSTM
│     ├── 04_prediction_alerting.ipynb    — BNN + MC Dropout
│     └── 05_perception_action_agent.ipynb — Agente FSM tiempo real
│
├── 📂 strategy-jose / strategy-II
│     Pipeline de detección multitarea con CLAHE uniforme (José Antonio)
│     ├── Preprocesamiento CLAHE uniforme sobre ~70,000 imágenes
│     ├── Fine-tuning YOLOv8n (mAP@0.5 = 0.573 · 97 épocas)
│     └── Agente ADAS heurístico >30 FPS · latencia ≈33ms
│
└── 📂 strategy-manuel / strategy-III
Pipeline con Retinex + CLAHE (Manuel)
├── Preprocesamiento CLAHE nocturno + Retinex diurno
├── Entrenamiento YOLOv8n 9 clases (50 épocas)
└── mAP@0.5 = 0.458 · mAP@0.5:0.95 = 0.254

> 💡 Cada rama tiene su propio README con instrucciones de instalación y reproducción de experimentos.

---

## 📊 Resultados Comparativos

### Percepción Visual (YOLOv8n — detección de objetos)

| Estrategia | Preprocesamiento | mAP@0.5 | mAP@0.5:0.95 | FPS |
|-----------|----------------|---------|-------------|-----|
| **I** (Roy) | Norm. nocturna + CLAHE diurna | 0.0886* | 0.0572 | 44–8.1 |
| **II** (José) ⭐ | CLAHE uniforme | **0.573** | **0.297** | >30 |
| **III** (Manuel) | CLAHE nocturna + Retinex diurna | 0.458 | 0.254 | — |

> *\*Bajo mAP intencional: en la Estrategia I, YOLOv8n actúa como **extractor de nodos para la GCN**, no como detector final — el mAP no es la métrica relevante para su rol.*

### Modelado de Riesgo Temporal (Estrategia I)

| Módulo | Conjunto | Pearson r | Métrica adicional |
|--------|---------|-----------|------------------|
| RiskLSTM | Validación | **0.916** | MAE = 0.145 |
| BNNRiskPredictor | Test (BDDA) | 0.615 | val_loss = 0.0148 |
| SegFormer-B0 | Validación | — | val_mIoU = 0.4661 |

### Conclusión del Estudio Comparativo

> La **Estrategia II** obtuvo el mejor mAP de detección (0.573), con 96% de sensibilidad en señales y 91% en vehículos, y es la arquitectura óptima para un sistema ADAS en tiempo real. La **Estrategia I** complementa este resultado con el modelado temporal más sofisticado (r=0.916) y cuantificación de incertidumbre bayesiana.

---

## ⚙️ Stack Tecnológico

- **Visión computacional:** YOLOv8n · SegFormer-B0 · ResNet-18+SPP
- **Deep Learning:** PyTorch 2.5.1 · PyTorch Geometric (GCN) · MC Dropout (BNN)
- **Hardware:** NVIDIA GTX 1660 Ti (6 GB VRAM) · CUDA 12.1
- **Entorno:** Python 3.10 · Jupyter Notebooks · Linux

---

## ⚠️ Estado del Proyecto

> Este repositorio se encuentra en **desarrollo activo** (*Work in Progress*).

- [x] Pipeline de 5 etapas completo y verificado (Estrategia I)
- [x] Validación OOD sobre BDDA (925 clips reales)
- [x] Estudio comparativo de 3 estrategias finalizado
- [x] Reporte de investigación redactado
- [ ] Presentación final Samsung Innovation Campus
- [ ] Exploración de deployment en Hailo-8 / Raspberry Pi 5

---

## 👥 Autores

Desarrollado como parte del programa **Samsung Innovation Campus 2025–2026**.

| Nombre | Institución | LinkedIn |
|--------|------------|----------|
| **Rogelio Leonardo Méndez Macías** | UAM Azcapotzalco | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rogelio-leonardo-mendez-macias/) |
| **José Antonio Coyotzi Juárez** | BUAP | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jos%C3%A9-antonio-coyotzi-ju%C3%A1rez-04047238b/) |
| **Manuel de Jesús Escobedo Sánchez** | UAZ | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/manuel-de-jes%C3%BAs-escobedo-s%C3%A1nchez-b0a7a9232/) |

**Supervisor:** M. en C. Eduardo De Ávila Armenta

---

<div align="center">
  <sub>Equipo 8 · Grupo 8 · Samsung Innovation Campus 2025–2026 · México</sub>
</div>


