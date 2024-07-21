# Un estudio comparativo de PCA y t-SNE en textos procesados con NLP

Este repositorio forma parte de un TFM del Máster de Ingeniería Matemática por la UCM.

El trabajo plantea un experimento de comparación de eficacia de dos técnicas de reducción de dimensionalidad: PCA y t-SNE. Este experimento se ha realizado sobre un corpus de *tweets* (textos cortos de la red social X, antes Twitter) procesados con técnicas de NLP. 

En el repositorio se almacenan:
- Datos: los datos en crudo descargados de la API de Twitter, resultados intermedios, resultados finales y gráficas explicativas.
- Códigos: código que procesa y estructura el corpus, código que aplica los modelos de reducción de dimensionalidad, códigos que evalúan los modelos y códigos que extraen las gráficas.
- Logs: los registros que informan del estado de las ejecuciones.

## Tabla de Contenidos
- [Instalación](#instalación)
- [Uso](#uso)
- [Capturas de Pantalla](#capturas-de-pantalla)
- [Contribuir](#contribuir)
- [Autores](#autores)
- [Licencia](#licencia)
- [Reconocimientos](#reconocimientos)

## Instalación

1. Asegúrate de tener Docker instalado. Puedes descargarlo e instalarlo desde [aquí](https://www.docker.com/get-started).

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

5. Crea un contenedor de docker:
    ```sh
    docker run --name TFM_repo -v PCA_TSNE_NLP_comparison:/opt/shared -p 8890:0001 -it python:3.11.8
    ```

3. Instala las dependencias:
    ```sh
    ./setup.sh
    ```

## Uso


## Contribuir


