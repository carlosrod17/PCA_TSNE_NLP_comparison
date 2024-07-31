En esta carpeta se encuentran los resultados de los 3 enfoques de evaluación de los modelos, que se han generado con los códigos 📄`code/Python/3_1_CLUSTERING_MODELS_WITH_K_FIXED.py`, 📄`code/Python/3_2_CLUSTERING_MODELS_WITH_K_OPTIMIZED.py` y 📄`code/Python/3_3_CLUSTERING_MODELS_DBSCAN.py`.

El **Enfoque 1** es el que aplica varios algoritmos de *clustering* con *k* fijado en 10 y contrasta la mejor clasificación obtenida de cada modelo con la clasificación original. Los resultados de este enfoque están en los archivos:

    📄 3_1_classification.csv.gz
    📄 3_1_metrics.csv
    📄 3_1_silhouette.csv

El **Enfoque 2** es el que aplica varios algoritmos de *clustering* con *k* fijado variando entre 2 y 30 y contrasta la mejor clasificación obtenida de cada modelo con la clasificación original. Los resultados de este enfoque están en los archivos:

    📄 3_2_classification.csv.gz
    📄 3_2_metrics.csv
    📄 3_2_silhouette.csv

El **Enfoque 3** es el que aplica el algoritmo DBSCAN y contrasta la clasificación obtenida de cada modelo con la clasificación original. Los resultados de este enfoque están en los archivos:

    📄 3_3_classification.csv.gz
    📄 3_3_metrics.csv

A continuación se presentan los esquemas de datos de estos archivos.

Los ficheros 📄`3_*_classification.csv.gz` tienen el mismo esquema. El esquema de datos es el siguiente:

    {
        "title": "3_*_classification.csv.gz",
        "type": "object",
        "properties": {
            "separator": {
                "type": "string",
                "enum": ["|"],
                "description": ""
            },
            "columns": {
                "type": "array",
                "items": [
                    {
                        "name": "TWEET_ID",
                        "type": "string",
                        "description": "Identifier"
                    },
                    {
                        "name": "CLUSTER",
                        "type": "string",
                        "description": ""
                    },
                    {
                        "name": "top_word_1",
                        "type": "string",
                        "description": "Most frequent word in tweet"
                    },
                    {
                        "name": "top_word_2",
                        "type": "string",
                        "description": "Second most frequent word in tweet"
                    },
                    {
                        "name": "top_word_3",
                        "type": "string",
                        "description": "Third most frequent word in tweet"
                    },
                    {
                        "name": "top_word_4",
                        "type": "string",
                        "description": "Forth most frequent word in tweet"
                    },
                    {
                        "name": "top_word_5",
                        "type": "string",
                        "description": "Fifth most frequent word in tweet"
                    },
                    {
                        "name": "CLUSTER_REAL",
                        "type": "string",
                        "description": ""
                    },
                    {
                        "name": "LIST_2",
                        "type": "string",
                        "description": "Tweet processed and formatted for display in an interactive chart"
                    },
                    {
                        "name": "DR_model",
                        "type": "string",
                        "description": "Dimensionality reduction model used"
                    },
                    {
                        "name": "x",
                        "type": "float",
                        "description": "First coordinate of the embedded point"
                    },
                    {
                        "name": "y",
                        "type": "float",
                        "description": "Second coordinate of the embedded point"
                    },
                    {
                        "name": "CLUSTER_TYPE",
                        "type": "string",
                        "description": "Indicates if the cluster becomes to the original classification"
                    },

                ],
                "additionalItems": false
            }
        },
        "required": ["separator", "columns"]
    }


