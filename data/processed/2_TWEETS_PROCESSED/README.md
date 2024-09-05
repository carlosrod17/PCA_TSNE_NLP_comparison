En esta carpeta se encuentra el resultado del procesamiento de textos y estructuración del corpus textual. En ella encontramos los ficheros 📄`tweets_preprocessed.csv.gz`, 📄`tweets_processed.csv.gz` y 📄`tweets_resampled.csv.gz` que contienen los resultados de los sucesivos pasos del procesamiento de textos, el fichero 📄`tweets_tfidf.csv.gz` que contiene la matriz de frecuencias derivada del corpus procesado y el fichero 📄`top_words.csv.gz` que contiene una lista de los *tokens* más representativos en los textos de cada clase original.

El esquema de datos de los ficheros 📄`tweets_preprocessed.csv.gz`, 📄`tweets_processed.csv.gz` y 📄`tweets_resampled.csv.gz` es el mismo. Este esquema es el siguiente:

    {
        "title": "tweets_processed.csv.gz",
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
                        "name": "CLUSTER_REAL",
                        "type": "string",
                        "description": ""
                    },
                    {
                        "name": "TWEET",
                        "type": "string",
                        "description": "Original text"
                    },
                    {
                        "name": "LIST_1",
                        "type": "string",
                        "description": "Result of first step of processing the text (tokenization, removing stopwords, urls, hastags and mentions)"
                    },
                    {
                        "name": "LIST_2",
                        "type": "string",
                        "description": "Result of second step of processing the text (langugage filterting)"
                    },
                    {
                        "name": "LIST_3",
                        "type": "string",
                        "description": "Result of third step of processing the text (tag filtering and lemmatizing)"
                    },
                    {
                        "name": "LIST_4",
                        "type": "string",
                        "description": "Result of fourth step of processing the text (stemming)"
                    },
                    {
                        "name": "LIST_5",
                        "type": "string",
                        "description": "Result of fifth step of processing the text (token embedding by similarity)"
                    },
                    {
                        "name": "LIST_6",
                        "type": "string",
                        "description": "Result of sixth step of processing the text (least frequent tokens removal)"
                    },
                ],
                "additionalItems": false
            }
        },
        "required": ["separator", "columns"]
    }

El fichero 📄`tweets_tfidf.csv.gz` contiene una matriz tfidf, donde cada fila se corresponde con un texto y cada columna con un *token* y los valores son *float*.

El fichero 📄`top_words.csv.gz` tiene el siguiente esquema:

    {
        "title": "top_words.csv.gz",
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
                        "name": "CLUSTER",
                        "type": "string",
                        "description": ""
                    },
                    {
                        "name": "top_word_1",
                        "type": "string",
                        "description": "Most frequent word in class"
                    },
                    {
                        "name": "top_word_2",
                        "type": "string",
                        "description": "Second most frequent word in class"
                    },
                    {
                        "name": "top_word_3",
                        "type": "string",
                        "description": "Third most frequent word in class"
                    },
                    {
                        "name": "top_word_4",
                        "type": "string",
                        "description": "Forth most frequent word in class"
                    },
                    {
                        "name": "top_word_5",
                        "type": "string",
                        "description": "Fifth most frequent word in class"
                    },
                ],
                "additionalItems": false
            }
        },
        "required": ["separator", "columns"]
    }

