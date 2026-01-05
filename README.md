# Práctica 2 – Tipología y Ciclo de Vida de los Datos

Este repositorio contiene el desarrollo completo de la **Práctica 2** de la asignatura **Tipología y Ciclo de Vida de los Datos**, correspondiente al **Máster Universitario en Ciencia de Datos** de la **Universitat Oberta de Catalunya (UOC)**.

El proyecto aborda de forma integrada las distintas fases del ciclo de vida del dato, desde su obtención hasta la extracción de conocimiento, aplicando técnicas de análisis de datos sobre un conjunto de datos del sector inmobiliario.

---

## Descripción del proyecto

El objetivo principal del proyecto es analizar la relación entre las características de los inmuebles y su **precio de venta**, con el fin de comprender qué factores influyen en su determinación y evaluar la capacidad explicativa de distintos modelos analíticos.

La base de datos inicialmente propuesta en la asignatura no contenía un número suficiente de variables numéricas para desarrollar adecuadamente los análisis requeridos. Por este motivo, se decidió obtener un nuevo conjunto de datos mediante **técnicas de web scraping**, aplicadas a una página web inmobiliaria de acceso público.

El dataset resultante incluye información estructurada sobre:

- Superficies  
- Número de habitaciones y baños  
- Localización  
- Precio de los inmuebles  

---

## Flujo de trabajo y ciclo de vida del dato

El desarrollo del proyecto sigue un flujo de trabajo alineado con las etapas clásicas del ciclo de vida del dato:

1. Obtención de los datos mediante web scraping y almacenamiento del dataset original sin modificar.  
2. Integración y selección de los datos, incluyendo el renombrado semántico de variables y la eliminación de atributos no relevantes.  
3. Limpieza y preparación de los datos, tratando valores faltantes, convirtiendo tipos de datos y homogeneizando unidades de medida.  
4. Análisis de los datos, que incluye:
   - Análisis exploratorio  
   - Modelado supervisado  
   - Modelado no supervisado  
   - Contraste de hipótesis  
5. Interpretación y comunicación de resultados en la memoria final.

---

## Estructura del repositorio

data/
├── raw/ # Datos originales obtenidos del scraping
├── processed/ # Datos tras integración, limpieza y selección

src/ # Código fuente del proyecto organizado por fases
figures/ # Gráficos generados durante el análisis
memoria/ # Memoria final de la práctica en formato PDF

requirements.txt # Dependencias necesarias para reproducir el análisis


---

## Descripción de los scripts

- **1_descripcion.py**  
  Carga del dataset original y análisis descriptivo inicial para comprender la estructura y tipología de las variables.

- **2_integracion_seleccion.py**  
  Integración semántica de las variables obtenidas mediante scraping, renombrado de columnas y selección de los atributos relevantes para el análisis.

- **3_limpieza.py**  
  Limpieza y acondicionamiento del dataset: tratamiento de valores faltantes, conversión de variables numéricas codificadas como texto, homogeneización de unidades de superficie y preparación de la variable precio.

- **4_analisis.py**  
  Aplicación de técnicas de análisis de datos: modelos de regresión para la estimación del precio, clustering para la segmentación de inmuebles y contraste de hipótesis para evaluar diferencias entre zonas geográficas.

- **5_graficos.py**  
  Generación de las visualizaciones utilizadas tanto en el análisis exploratorio como en la memoria final.

---

## Reproducibilidad

El proyecto ha sido desarrollado íntegramente en **Python**.

Para reproducir el análisis es necesario instalar las dependencias indicadas en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt

Observaciones finales

Este repositorio tiene una finalidad exclusivamente académica y forma parte de la evaluación continua de la asignatura Tipología y Ciclo de Vida de los Datos.
