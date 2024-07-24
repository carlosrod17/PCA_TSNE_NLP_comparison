En esta carpeta se encuentra un solo fichero 📄`tfidf_embedded.csv.gz` con la información de los datos proyectados por los diferentes modelos de reducción de dimensionalidad.

El esquema de datos de 📄`tfidf_embedded.csv.gz` es el siguiente:

    {
        "title": "tfidf_embedded.csv.gz",
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

                ],
                "additionalItems": false
            }
        },
        "required": ["separator", "columns"]
    }
