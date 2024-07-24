# Un estudio comparativo de PCA y t-SNE en textos procesados con NLP

Este repositorio forma parte de un TFM del Máster de Ingeniería Matemática por la UCM.

El trabajo plantea un experimento de comparación de eficacia de dos técnicas de reducción de dimensionalidad: PCA y t-SNE. Este experimento se ha realizado sobre un corpus de *tweets* (textos cortos de la red social X, antes Twitter) procesados con técnicas de NLP. 

## 📋 Tabla de Contenidos
- [🧩 Estructura](#estructura)
- [📦 Instalación](#instalación)
- [🚀 Uso](#uso)
- [💡 Contribuir](#contribuir)

## 🧩 Estructura

Este repositorio tiene la siguiente estructura:

    📁 PCA_TSNE_NLP_comparison

       📁 code                              # Contiene los códigos R y python y la configuración
       📁 data                              # Contiene todos los datos,resultados y gráficas
       📁 logs                              # Contiene los registros de las ejecuciones

       🐙 .gitignore                        # Lista de los ficheros que no requieren control de versiones
       🚀 executions.sh                     # Orquestador de las ejecuciones
       📄 README.md                         # Este archivo
       📦 requirements.txt                  # Lista de las dependencias de python
       📦 setup.sh                          # Ejecutable de instalación
       📘TFM_CarlosRodrigoPascual.pdf       # Memoria del TFM


La carpeta 📁`code` tiene el siguiente contenido:

    📁 code
       📄 __init__.py                                           # Archivo auxiliar  
       📄 0_download_tweets.R                                   # Código en R        
       📄 1_PROCESSING.py                                       # Código en Python
       📄 2_DIMENSIONALITY_REDUCTION.py                         # Código en Python
       📄 3_1_CLUSTERING_MODELS_WITH_K_FIXED.py                 # Código en Python
       📄 3_2_CLUSTERING_MODELS_WITH_K_OPTIMIZED.py             # Código en Python
       📄 3_3_DBSCAN_MODELS.py                                  # Código en Python
       📄 4_1_GET_INTERACTIVE_FIGURES.py                        # Código en Python
       📄 4_2_GET_TFM_FIGURES.py                                # Código en Python
       📄 functions.py                                          # Código en Python
       📄 var_def.py                                            # Código en Python

       
La carpeta 📁`data` tiene el siguiente contenido:

    📁 data
       📁 auxiliar                                  # Contiene datos auxiliares 
       📁 figures
       📁 figures_TFM
       📁 processed                                
          📁 1_TWEETS_RAW_BY_CLUSTER                # Contiene los datos en crudo
          📁 2_TWEETS_PROCESSED                     # Contiene el corpus procesado
          📁 3_TWEETS_EMBEDDED                      # Contiene el corpus proyectado
          📁 4_RESULTS                              # Contiene los resultados

Las carpetas dentro de 📁`data/processed` contienen sus respectivos 📄`README.md` detallando el contenido y el *Schema* de los datos.


## 📦 Instalación

1. Asegúrate de tener 🐋Docker instalado. Puedes descargarlo e instalarlo desde [aquí](https://www.docker.com/get-started).

2. Descargate la imagen de python:
    ```sh
    docker pull python:3.11.8
    ```

3. Clona el repositorio:
    ```sh
    git clone https://github.com/carlosrod17/PCA_TSNE_NLP_comparison.git
    ```

4. Navega al directorio del proyecto:
    ```sh
    cd PCA_TSNE_NLP_comparison
    ```

5. Crea un contenedor de 🐋Docker:
    ```sh
    docker run --name TFM_repo -v PCA_TSNE_NLP_comparison:/opt/shared -p 8890:0001 -it python:3.11.8
    ```

3. Instala las dependencias:
    ```sh
    ./setup.sh
    ```

## 🚀 Uso

1. Modificar el fichero 📄`/opt/shared/code/var_def.py` para configurar la ejecución.
2. Modificar el orquestador 🚀`/opt/shared/executions.sh` para escoger los códigos a ejecutar. 

    ⚠️ **Los códigos deben ejecutarse en el orden presentado**

3. Monitorizar la ejecución con el archivo .log generado en la carpeta 📁`/opt/shared/logs`.        

## 💡Contribuir

Las contribuciones son bienvenidas. Por favor, sigue estos pasos para contribuir:

1. Haz un fork del repositorio.
2. Crea una rama para tu contribución (`git checkout -b mi-nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -am 'Añadir nueva funcionalidad'`).
4. Envía tus cambios (`git push origin mi-nueva-funcionalidad`).
5. Abre un pull request describiendo tus cambios.