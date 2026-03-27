# 🚗 Accident Detection System
### Samsung Innovation Campus - Equipo 8 (Grupo 8)

---

## 📈 Reporte Ejecutivo: Hitos de la Semana 1

Durante la fase inicial del proyecto, se establecieron los cimientos técnicos de datos y se validó la viabilidad del modelo predictivo. Los principales resultados obtenidos son:

- **Ingeniería de Datos (Data Engineering):** Limpieza, estandarización y procesamiento exitoso de **61,728 muestras** provenientes del dataset BDD100K, consolidando una base de datos estructurada y de alta calidad.
- **Análisis Exploratorio de Datos (EDA):** Identificación de patrones críticos en la distribución de condiciones climáticas y franjas horarias, fundamentales para la inferencia de factores de riesgo vial.
- **Modelo Baseline:** Entrenamiento y validación de un algoritmo *Random Forest* que alcanzó una precisión (*Accuracy*) de **1.0**, comprobando matemáticamente la validez de la lógica de evaluación de riesgo ambiental propuesta.
- **Infraestructura para Deep Learning:** Generación y normalización automatizada de etiquetas (formato `.txt`), habilitando la transición hacia arquitecturas de detección de objetos (YOLO) en las siguientes fases.

---

## 📂 Estructura del Proyecto

Para mantener el repositorio ligero y eficiente, hemos separado la lógica del proyecto de los activos de datos pesados.
```text
object_accident_detection/
├── .gitignore                  # Reglas para ignorar archivos pesados
├── README.md                   # Documentación principal del proyecto
├── requirements.txt            # Dependencias necesarias (pip install -r)
├── notebooks/                  # Desarrollo de modelos y análisis
│   └── S1_data_engineering.ipynb
├── src/                        # Scripts auxiliares y utilitarios
├── env_samsung/                # [Ignorado] Entorno virtual local
└── data/                       # [Ignorado - Drive] Almacenamiento de datos
    ├── processed/              # CSVs con 61,728 registros procesados
    └── dataset_bdd100k/        # Dataset Berkeley DeepDrive
        └── train/
            └── labels_yolo.zip # Etiquetas .txt listas para YOLOv8
```

---

## 🛠️ Guía de Configuración para el Equipo

Sigue estos pasos para sincronizar tu entorno de trabajo:

### 1. Clonar el repositorio
```bash
git clone https://github.com/Rogelio756/Equipo8-Grupo8-SIC-2025-.git
git checkout s1
```

### 2. Configurar el entorno virtual
```bash
python -m venv env_samsung

# Activar en Windows:
.\env_samsung\Scripts\activate

# Instalar librerías:
pip install -r requirements.txt
```

### 3. Descarga de datos (Google Drive)

Debido al volumen de datos (60k+ archivos), los activos pesados se gestionan en la nube.

📂 [Acceso al Repositorio de Datos (Google Drive)](#)

> **Instrucción:** Descarga el contenido completo de la carpeta de Drive y colócalo en la raíz del proyecto dentro de un directorio llamado `data/`. Esto garantizará la integridad de las rutas relativas al ejecutar los notebooks.
