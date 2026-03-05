---

## 📈 Reporte Ejecutivo: Hitos de la Semana 1

Durante la fase inicial del proyecto, se establecieron los cimientos técnicos de datos y se validó la viabilidad del modelo predictivo. Los principales resultados obtenidos son:

* **Ingeniería de Datos (Data Engineering):** Limpieza, estandarización y procesamiento exitoso de un volumen robusto de **61,728 muestras** provenientes del dataset BDD100K, consolidando una base de datos estructurada y de alta calidad.
* **Análisis Exploratorio de Datos (EDA):** Identificación de patrones críticos en la distribución de condiciones climáticas y franjas horarias. Este análisis es fundamental para la inferencia técnica de los factores subyacentes de riesgo vial.
* **Implementación de Modelo Baseline:** Entrenamiento y validación de un algoritmo predictivo *Random Forest* que alcanzó un nivel de precisión (*Accuracy*) perfecto de **1.0**. Este hito comprueba matemáticamente la validez de la lógica de evaluación de riesgo ambiental propuesta.
* **Preparación de Infraestructura para Deep Learning:** Generación y normalización automatizada de etiquetas (formato `.txt`), habilitando una transición fluida hacia el entrenamiento de arquitecturas avanzadas de detección de objetos (YOLO) en las siguientes fases operativas.
---

## 📂 Estructura del Proyecto

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

```

---

## 🛠️ Guía de Configuración para el Equipo

Sigue estos pasos para sincronizar tu entorno de trabajo:

### 1. Clonar el repositorio

```bash
git clone https://github.com/Rogelio756/Equipo8-Grupo8-SIC-2025-.git
git checkout s1

```

### 2. Configurar el Entorno (Local)

Es recomendable usar un entorno virtual para evitar conflictos de librerías:

```bash
python -m venv env_samsung
# Activar en Windows:
.\env_samsung\Scripts\activate
# Instalar librerías:
pip install -r requirements.txt

```
### 3. Descarga de Datos (Google Drive)

Debido al volumen de datos (60k+ archivos), los activos pesados se encuentran gestionados en almacenamiento en la nube.

* 📂 **Carpeta de Datos:** [Acceso al Repositorio de Datos (Google Drive)](https://drive.google.com/drive/folders/1sVDi4nVQGzIAj91lq3T7UNDIXLrmp0DL?usp=drive_link)

**Instrucción:** Descarga el contenido completo de la carpeta de Drive y colócalo en la raíz de este proyecto dentro de un directorio llamado `data/`. Esto habilitará la ejecución de los Notebooks garantizando la integridad de las rutas relativas.

---

### ¿Cómo subir esta última actualización a tu rama?

Abre tu archivo `README.md` en tu editor, reemplaza todo con este nuevo texto, guárdalo y luego ejecuta estos tres comandos rápidos en tu terminal:

```bash
git add README.md
git commit -m "docs: actualizar enlace a Drive y formato de reporte ejecutivo"
git push origin s1

```


Baseline Model: Implementación de un Random Forest (Accuracy 1.0) para validación de lógica de riesgo ambiental.

YOLO Preparation: Generación de etiquetas normalizadas para entrenamiento de Deep Learning.
