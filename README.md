# 🚗 Accident Detection System: Machine Learning Capstone
**Samsung Innovation Campus — Equipo 8 | Grupo 8**

Sistema inteligente de detección de objetos y prevención de accidentes viales. Utiliza Aprendizaje Supervisado y Visión Artificial para identificar riesgos críticos mediante análisis de contexto y percepción visual. Solución técnica enfocada en la seguridad vial y la toma de decisiones automatizada en tiempo real.

---

## 📂 Estructura del Proyecto

Para mantener el repositorio ligero y eficiente, hemos separado la lógica del proyecto de los activos de datos pesados.
```text
object_accident_detection/
├── .gitignore                   # Reglas para ignorar archivos pesados
├── README.md                    # Documentación principal del proyecto
├── requirements.txt             # Dependencias necesarias (pip install -r)
├── notebooks/                   # Desarrollo de modelos y análisis
│   └── S1_data_engineering.ipynb
├── src/                         # Scripts auxiliares y utilitarios
├── env_samsung/                 # [Ignorado] Entorno virtual local
└── data/                        # [Ignorado - Drive] Almacenamiento de datos
    ├── processed/               # CSVs con 61,728 registros procesados
    └── dataset_bdd100k/         # Dataset Berkeley DeepDrive
        └── train/
            └── labels_yolo.zip  # Etiquetas .txt listas para YOLOv8
```

---

## 🧠 Preprocessing Pipeline

Se diseñó y validó un pipeline de preprocesamiento de imágenes enfocado en mejorar la calidad de entrada para modelos de visión (YOLO / CNNs) en escenarios de conducción real.

📓 Notebook: `notebooks/01_preprocessing.ipynb`

### ⚙️ Pipeline Final
```
Imagen → CLAHE (condicional) → Resize (416×416) → Normalización (ImageNet)
```

### 🔹 Componentes

**1. CLAHE (Adaptive Contrast Enhancement)**

Aplicado solo en condiciones adversas:
- 🌙 Night
- 🌧️ Rainy / Foggy / Snowy

Mejora el contraste local y la visibilidad en entornos difíciles.

---

**2. Resize**

Resolución seleccionada: **416×416**

Balance óptimo entre precisión y velocidad. Compatible con modelos YOLO.

---

**3. Normalización (ImageNet)**
```python
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

Alinea la distribución del dataset con modelos preentrenados, mejorando estabilidad y convergencia.

---

### 🔬 Experimentos Realizados

| # | Experimento | Resultado |
|---|-------------|-----------|
| 1 | CLAHE vs Retinex | CLAHE superior; Retinex inestable |
| 2 | Pipeline adaptativo (CLAHE condicional + normalización) | Mejora en condiciones adversas sin degradar condiciones óptimas |
| 3 | Resolución: 320 vs **416** vs 640 | 416 ofrece el mejor equilibrio rendimiento/eficiencia |

### 🎯 Conclusión

- CLAHE mejora significativamente imágenes nocturnas
- La normalización estabiliza la entrada del modelo
- El pipeline es adaptativo y robusto
- La resolución **416×416** ofrece el mejor compromiso entre rendimiento y eficiencia

---

## 🛠️ Configuración del Entorno

**1. Clonar el repositorio**
```bash
git clone https://github.com/Rogelio756/Equipo8-Grupo8-SIC-2025-.git
git checkout s1
```

**2. Crear y activar el entorno virtual**
```bash
python -m venv env_samsung

# Windows
.\env_samsung\Scripts\activate

# macOS / Linux
source env_samsung/bin/activate
```

**3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

---

## 📈 Estado del Proyecto

| Módulo | Estado |
|--------|--------|
| Data Engineering (61,728 muestras BDD100K) | ✅ Completado |
| EDA — Análisis climático y temporal | ✅ Completado |
| Baseline Model (Random Forest) | ✅ Completado |
| YOLO Preparation — Generación de etiquetas | ✅ Completado |
| Preprocessing Pipeline | ✅ Completado |
| Entrenamiento de modelos YOLO | 🔜 Próximo paso |

---

## 🚀 Próximo Paso

Entrenamiento de modelos de detección (YOLO) utilizando el pipeline validado.
