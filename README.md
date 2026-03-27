# 🚗 Accident Detection System: Machine Learning Capstone
Samsung Innovation Campus - Equipo 8 (Grupo 8)

Sistema inteligente de detección de objetos y prevención de accidentes viales. Utiliza Aprendizaje Supervisado y Visión Artificial para identificar riesgos críticos mediante el análisis de contexto y percepción visual. Solución técnica enfocada en la seguridad vial y la toma de decisiones automatizada en tiempo real.

---

# 📂 Estructura del Proyecto

Para mantener el repositorio ligero y eficiente, hemos separado la lógica del proyecto de los activos de datos pesados.


```text
object_accident_detection/
├── .gitignore                 # Reglas para ignorar archivos pesados
├── README.md                  # Documentación principal del proyecto
├── requirements.txt           # Dependencias necesarias (pip install -r)
├── notebooks/                 # Desarrollo de modelos y análisis
│   └── S1_data_engineering.ipynb
├── src/                       # Scripts auxiliares y utilitarios
│
├── env_samsung/               # [Ignorado] Entorno virtual local
└── data/                      # [Ignorado - Drive] Almacenamiento de datos
    ├── processed/             # CSVs con 61,728 registros procesados
    └── dataset_bdd100k/       # Dataset Berkeley DeepDrive
        └── train/
            └── labels_yolo.zip # Etiquetas .txt listas para YOLOv8

---

# 🧠 🔬 Preprocessing Pipeline (NUEVO)

Se diseñó y validó un pipeline de preprocesamiento de imágenes enfocado en mejorar la calidad de entrada para modelos de visión (YOLO / CNNs) en escenarios de conducción real.

📓 Notebook: `notebooks/01_preprocessing.ipynb`

---

## ⚙️ Pipeline Final

---
Imagen → CLAHE (condicional) → Resize (416×416) → Normalización (ImageNet)
## 🔹 Componentes

### 1. CLAHE (Adaptive Contrast Enhancement)
- Aplicado solo en condiciones adversas:
  - 🌙 night
  - 🌧️ rainy / foggy / snowy
- Mejora contraste local y visibilidad

---

### 2. Resize
- Resolución seleccionada: **416×416**
- Balance entre:
  - precisión
  - velocidad
- Compatible con modelos YOLO

---

### 3. Normalización (ImageNet)
Imagen → CLAHE (condicional) → Resize (416×416) → Normalización (ImageNet)
Valores utilizados:
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


✔️ Justificación:
- Alinea la distribución del dataset con modelos preentrenados
- Mejora estabilidad y convergencia

---

## 🔬 Experimentos Realizados

### 🔹 Experimento 1 — Filtros
- CLAHE vs Retinex
- Resultado: CLAHE superior, Retinex inestable

---

### 🔹 Experimento 2 — Pipeline adaptativo
- CLAHE condicional + normalización
- Resultado: mejora en condiciones adversas sin degradar condiciones óptimas

---

### 🔹 Experimento 3 — Resolución (Resize)
- Evaluación: 320 vs 416 vs 640
- Resultado:
  - 320 → pérdida de detalles
  - 640 → mayor costo computacional
  - **416 → mejor equilibrio**

---

## 🎯 Conclusión del Preprocessing

- CLAHE mejora significativamente imágenes nocturnas
- La normalización estabiliza la entrada del modelo
- El pipeline es adaptativo y robusto
- La resolución 416×416 ofrece el mejor compromiso entre rendimiento y eficiencia

---

# 🛠️ Guía de Configuración para el Equipo

## 1. Clonar el repositorio

```bash
git clone https://github.com/Rogelio756/Equipo8-Grupo8-SIC-2025-.git
git checkout s1

2. Configurar el Entorno
python -m venv env_samsung

# Activar en Windows:
.\env_samsung\Scripts\activate

pip install -r requirements.txt

📈 Avances
✔️ Data Engineering
61,728 muestras procesadas (BDD100K)
✔️ EDA
Análisis climático y temporal
✔️ Baseline Model
Random Forest (validación lógica)
✔️ YOLO Preparation
Generación de etiquetas
🔥 Preprocessing Pipeline (NUEVO)
Pipeline validado listo para entrenamiento
Mejora de imágenes + estandarización
🚀 Próximo Paso

Entrenamiento de modelos de detección (YOLO) utilizando el pipeline validado.

---

# 🚀 Listo

Solo:

1. Pega esto en tu `README.md`
2. Guarda
3. Haz:

```bash
git add README.md
git commit -m "docs: add preprocessing pipeline section"
git push origin preprocessing
