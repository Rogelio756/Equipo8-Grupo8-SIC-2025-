🚗 Accident Detection System: Machine Learning Capstone
Samsung Innovation Campus - Equipo 8 (Grupo 8)
Sistema inteligente de detección de objetos y prevención de accidentes viales. Utiliza Aprendizaje Supervisado y Visión Artificial para identificar riesgos críticos mediante el análisis de contexto y percepción visual. Solución técnica enfocada en la seguridad vial y la toma de decisiones automatizada en tiempo real.

📂 Estructura del Proyecto
Para mantener el repositorio ligero y eficiente, hemos separado la lógica del proyecto de los activos de datos pesados.

Plaintext
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
🛠️ Guía de Configuración para el Equipo
Sigue estos pasos para sincronizar tu entorno de trabajo:

1. Clonar el repositorio
Bash
git clone https://github.com/Rogelio756/Equipo8-Grupo8-SIC-2025-.git
git checkout s1
2. Configurar el Entorno (Local)
Es recomendable usar un entorno virtual para evitar conflictos de librerías:

Bash
python -m venv env_samsung
# Activar en Windows:
.\env_samsung\Scripts\activate
# Instalar librerías:
pip install -r requirements.txt
3. Descarga de Datos (Google Drive)
Debido al volumen de datos (60k+ archivos), los activos pesados se encuentran en Google Drive.

📂 Carpeta de Datos: [PEGA AQUÍ TU LINK DE DRIVE]

Instrucción: Descarga la carpeta data/ completa y colócala en la raíz de este proyecto. Esto habilitará la ejecución de los Notebooks sin necesidad de cambiar rutas de acceso.

📈 Avances de la Semana 1
Data Engineering: Procesamiento de 61,728 muestras del dataset BDD100K.

Exploratory Data Analysis (EDA): Análisis de distribución climática y horaria.

Baseline Model: Implementación de un Random Forest (Accuracy 1.0) para validación de lógica de riesgo ambiental.

YOLO Preparation: Generación de etiquetas normalizadas para entrenamiento de Deep Learning.