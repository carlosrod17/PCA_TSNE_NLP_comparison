# Un estudio comparativo de PCA y t-SNE en textos procesados con NLP

Este repositorio forma parte de un Trabajo de Fin de Máster del Máster de Ingeniería Matemática por la UCM. El archivo 📘`TFM_CarlosRodrigoPascual.pdf` contiene la memoria que se presentó como resultado final del proyecto. El archivo 📕`TFM_CarlosRodrigoPascual_presentacion.pdf` contiene el soporte audiovisual utilizado durante la defensa.

El trabajo plantea un experimento de comparación de eficacia de dos técnicas de reducción de dimensionalidad: PCA y t-SNE. Este experimento se ha realizado sobre un corpus de *tweets* (textos cortos de la red social X, antes Twitter) procesados con técnicas de NLP. Para ello se ha utilizado un código de R para realizar la descarga de datos de la API de *Twitter* y varios códigos de Python con librerías como Spacy, NLTK, Sklearn, Plotly para procesar los datos, aplicarles los modelos, clasificar los datos, evaluar los modelos y graficar los resultados.

En este archivo se da una pequeña introducción del contenido del repositorio. Los contenidos son los siguientes:
- [🧩 Estructura](#-estructura)
- [📋 Requititos](#-requisitos)
- [📦 Instalación](#-instalación)
- [🚀 Uso](#-uso)

## 🧩 Estructura

Este repositorio tiene la siguiente estructura:

    📁 PCA_TSNE_NLP_comparison
       📁 code                                                  # Contiene los códigos R y Python y la configuración
       📁 data                                                  # Contiene todos los datos,resultados y gráficas
       📁 logs                                                  # Contiene los registros de las ejecuciones
       🐙 .gitignore                                            # Lista de los ficheros que no requieren control de versiones
       🚀 executions.sh                                         # Orquestador de las ejecuciones
       📄 README.md                                             # Este archivo
       📦 requirements.txt                                      # Lista de las dependencias de python
       📦 setup.sh                                              # Ejecutable de instalación
       📘 TFM_CarlosRodrigoPascual.pdf                          # Memoria del TFM
       📕 TFM_CarlosRodrigoPascual_presentacion.pdf             # Presentación del TFM


La carpeta 📁`code` tiene el siguiente contenido:

    📁 code
       📁 Python
          📁 utils                                              
             📄 __init__.py                                     
             📄 clustering.py                                   # Útiles de clustering
             📄 dimensionality_reduction.py                     # Útiles de reducción de dimensionalidad
             📄 natural_language_processing.py                  # Útiles de NLP 
          📄 __init__.py                                        
          📄 1_PROCESSING.py                                    # Código para el procesado del corpus
          📄 2_DIMENSIONALITY_REDUCTION.py                      # Código para la reducción de dimensionalidad
          📄 3_1_CLUSTERING_MODELS_WITH_K_FIXED.py              # Código para la clasificación (Enfoque 1)
          📄 3_2_CLUSTERING_MODELS_WITH_K_OPTIMIZED.py          # Código para la clasificación (Enfoque 2)
          📄 3_3_DBSCAN_MODELS.py                               # Código para la clasificación (Enfoque 3)
          📄 4_1_GET_INTERACTIVE_FIGURES.py                     # Código para la extracción de gráficas (html)
          📄 4_2_GET_TFM_FIGURES.py                             # Código para la extracción de gráficas (png)
          📄 var_def.py                                         # Código de configuración
       📁 R
          📄 0_download_tweets.R                                # Código para la descarga de datos        

       
La carpeta 📁`data` tiene el siguiente contenido:

    📁 data
       📁 auxiliar                                              # Contiene datos auxiliares 
       📁 figures
          📁 HTML                                               # Gráficas interactivas
          📁 PNG                                                # Gráficas estáticas
       📁 processed                                
          📁 1_TWEETS_RAW_BY_CLUSTER                            # Contiene los datos en crudo
          📁 2_TWEETS_PROCESSED                                 # Contiene el corpus procesado
          📁 3_TWEETS_EMBEDDED                                  # Contiene el corpus proyectado
          📁 4_RESULTS                                          # Contiene los resultados

Las carpetas dentro de 📁`data/processed` contienen sus respectivos 📄`README.md` detallando el contenido y el *Schema* de los datos.

## 📋 Requisitos

1. Tener instalado un entorno 🐧Linux o una alternativa para usar los scripts de Bash 📦`setup.sh` y 🚀`executions.sh` de instalación y ejecución.

2. Tener instalado 🐋Docker para simular correctamente el entorno del repositorio. 


## 📦 Instalación

1. Descargar la imagen de python:
    ```sh
    docker pull python:3.11.8
    ```

2. Clonar el repositorio:
    ```sh
    git clone https://github.com/carlosrod17/PCA_TSNE_NLP_comparison.git
    ```

3. Navegar al directorio del proyecto:
    ```sh
    cd PCA_TSNE_NLP_comparison
    ```

4. Crear un contenedor de 🐋Docker:
    ```sh
    docker run --name PCA_TSNE_container -v PCA_TSNE_NLP_comparison:/opt/shared -p 8890:0001 -it python:3.11.8
    ```

5. Instalar las dependencias:
    ```sh
    ./setup.sh
    ```

## 🚀 Uso

1. Encender el contenedor de 🐋Docker:
    ```sh
    docker start PCA_TSNE_container
    ```
2. Modificar el fichero 📄`code/var_def.py` para configurar la ejecución.
3. Modificar el orquestador 🚀`executions.sh` para escoger los códigos a ejecutar. 

    ⚠️ LOS CÓDIGOS DEBEN EJECUTARSE EN EL ORDEN PRESENTADO

4. Monitorizar la ejecución con el archivo .log generado en la carpeta 📁`logs`.
5. Revisar los ficheros generados en cada proceso (📁`data/processed`).
6. Al acabar, apagar el contenedor de 🐋Docker:
    ```sh
    docker stop PCA_TSNE_container
    ```