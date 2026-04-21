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

El documento cubre: introducción y problemática, estado del arte, metodología de las tres estrategias, resultados experimentales, discusión comparativa y trabajo futuro.

---

## 🔬 Estrategias de Pipeline

### 🔵 Estrategia I — Normalización Selectiva por Condición Lumínica *(Rogelio)*

El pipeline más complejo del estudio. Aplica **CLAHE condicional**: normalización de intensidad para imágenes nocturnas y CLAHE para imágenes diurnas. Sobre este preprocesamiento construye una arquitectura de **cinco etapas** en cascada: percepción multimodal (YOLOv8n + SegFormer-B0 + ResNet-18+SPP), modelado de grafos de interacción vehicular (GCNEncoder), predicción de series temporales (RiskLSTM) y cuantificación de incertidumbre bayesiana (BNN con MC Dropout, T=20 pasadas Monte Carlo). El agente final opera entre **44 FPS** en escenarios de riesgo bajo y **8.1 FPS** con el pipeline completo activo, sobre una GPU de 6 GB VRAM. Validado sobre 925 clips dashcam reales del conjunto BDDA.

**Métricas clave:** RiskLSTM Pearson r=0.916 · BNN Pearson r=0.615 (test) · SegFormer val_mIoU=0.4661 · YOLOv8n mAP@0.5=0.0886*

> *\*El bajo mAP es intencional: YOLOv8n actúa como extractor de nodos para la GCN, no como detector final.*

---

### 🟢 Estrategia II — Preprocesamiento Uniforme y Multitarea *(José Antonio)* ⭐ Mejor detección

Aplica **CLAHE uniforme** a la totalidad del dataset (~70,000 imágenes) sin distinción de condición lumínica, simplificando el preprocesamiento. Entrena YOLOv8n con **fine-tuning completo** sobre BDD100K en 4 macro-clases estratégicas (Vehículo, Señales, Peatón, Dos_Ruedas), con Early Stopping que detuvo el entrenamiento en la época 97 (mejor época: 82). Complementa la detección con YOLOv8n-seg preentrenado para máscaras de segmentación holográficas. El agente ADAS resultante procesa video real de forma fluida a más de **30 FPS** con latencia constante de ≈33 ms, incluso en escenas urbanas saturadas.

**Métricas clave:** mAP@0.5=**0.573** · Señales 96% sensibilidad · Vehículos 91% sensibilidad · >30 FPS latencia ≈33ms

---

### 🟠 Estrategia III — Mejora de Visibilidad mediante Retinex y CLAHE *(Manuel)*

Aplica tratamiento inverso según condición lumínica: **CLAHE para imágenes nocturnas** y el algoritmo de **Retinex para imágenes diurnas**, buscando restauración de color y corrección de alto brillo. Entrena YOLOv8n en **9 categorías** de tráfico (car, truck, bus, motorcycle, bicycle, person, train, traffic light, traffic sign) durante 50 épocas a resolución 640px. El análisis por clase revela un buen desempeño en vehículos (AP=0.62) y señalética (traffic light 0.47, traffic sign 0.45), con el mayor reto en la clase `person` (AP=0.27), frecuentemente confundida con el fondo.

**Métricas clave:** mAP@0.5=0.458 · mAP@0.5:0.95=0.254 · car AP=0.62 · person AP=0.27

---

## 📊 Resultados Comparativos

### Percepción Visual (YOLOv8n)

| Estrategia | Preprocesamiento | mAP@0.5 | mAP@0.5:0.95 | FPS |
|-----------|----------------|---------|-------------|-----|
| **I** (Rogelio) | Norm. nocturna + CLAHE diurna | 0.0886* | 0.0572 | 44 – 8.1 |
| **II** (José) ⭐ | CLAHE uniforme | **0.573** | **0.297** | >30 |
| **III** (Manuel) | CLAHE nocturna + Retinex diurna | 0.458 | 0.254 | — |

### Modelado de Riesgo Temporal (Estrategia I)

| Módulo | Conjunto | Pearson r | Métrica adicional |
|--------|---------|-----------|------------------|
| RiskLSTM | Validación | **0.916** | MAE = 0.145 |
| BNNRiskPredictor | Test (BDDA) | 0.615 | val_loss = 0.0148 |
| SegFormer-B0 | Validación | — | val_mIoU = 0.4661 |

> La **Estrategia II** logró el mejor rendimiento de detección (mAP@0.5=0.573) y es la arquitectura óptima para un sistema ADAS en tiempo real. La **Estrategia I** aporta el modelado temporal más sofisticado (r=0.916) con cuantificación de incertidumbre bayesiana.

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

