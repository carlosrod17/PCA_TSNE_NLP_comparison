En esta carpeta se encuentran los datos en crudo descargados de la API de X (antes *Twitter*) mediante el código 📄`code/0_download_tweets.R`. Estos datos están separados en 10 archivos 📄`tweets_*.csv`, uno para cada query múltiple lanzada. 

El esquema de datos de 📄`tweets_*.csv` es el siguiente:

    {
        "title": "tweets_i.csv",
        "type": "object",
        "properties": {
            "separator": {
                "type": "string",
                "enum": [";"],
                "description": "Column separator used in the CSV file"
            },
            "columns": {
                "type": "array",
                "items": [
                    {
                        "name": "",
                        "type": "number",
                        "description": "Identifier"
                    },
                    {
                        "name": "",
                        "type": "string",
                        "description": "Tweet"
                    }
                ],
                "additionalItems": false
            }
        },
        "required": ["separator", "columns"]
    }
