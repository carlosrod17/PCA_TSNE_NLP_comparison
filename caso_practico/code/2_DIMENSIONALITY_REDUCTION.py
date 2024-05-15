##############################################################################
# DIMENSIONALITY REDUCTION
##############################################################################

import os
import sys
import logging
import datetime
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from var_def import TFM_PATH
from var_def import BBDD_PATH_2
from var_def import BBDD_PATH_3
from var_def import FIGURES_PATH

from var_def import delimiter
from var_def import n_clusters

from var_def import DR_models_to_perform
from var_def import DR_models

from var_def import my_palette

from functions import fit_transform_model

logging.basicConfig(
    filename = os.path.join(TFM_PATH, sys.argv[1]),
    format="%(asctime)s %(levelname)s: %(message)s",
    level = logging.INFO)

logging.info("-"*120)
logging.info("Starting dimensionality reduction.")

n_DR = len(DR_models_to_perform)

logging.info("Loading tfidfs.")

DF_tfidf = (
    pd.read_csv(
        os.path.join(BBDD_PATH_2, "tweets_tfidf.csv.gz"),
        sep = delimiter,
        compression = "gzip",
        header = 0,
        index_col = "TWEET_ID"
    )
    .astype(float)
)

logging.info(f"{DF_tfidf.shape[0]:06d} tfidfs have being loaded.")

# PREPARATION

DF_aux = (
    pd.merge(
        pd.read_csv(
            os.path.join(BBDD_PATH_2, "top_words.csv.gz"),
            sep = delimiter,
            compression = "gzip", 
            header = 0,
            dtype = str
        ),
        pd.read_csv(
            os.path.join(BBDD_PATH_2, "tweets_resampled.csv.gz"),
            sep = delimiter,
            compression = "gzip",
            header = 0,
            usecols = ["TWEET_ID", "CLUSTER_REAL", "LIST_2"],
            dtype = str
        ),
        left_on = "CLUSTER",
        right_on = "CLUSTER_REAL"
    )
    .set_index("TWEET_ID")
)

# print(DF_aux.head())

# IMPLEMENTATION

DF_tfidf_embedded = pd.DataFrame(None)

for model in DR_models_to_perform:
    
    logging.info(f"Performing {model} model.")
    
    MAT_tfidf_embedded = fit_transform_model(data = DF_tfidf,
                                             model = model,
                                             params = DR_models[model]["parameters"])
    
    DF_tfidf_embedded = pd.concat(
        [
            DF_tfidf_embedded,
            DF_aux
            .assign(
                DR_model = model,
                x = (MAT_tfidf_embedded[:,0] - MAT_tfidf_embedded[:,0].min())/(MAT_tfidf_embedded[:,0].max() - MAT_tfidf_embedded[:,0].min()),
                y = (MAT_tfidf_embedded[:,1] - MAT_tfidf_embedded[:,1].min())/(MAT_tfidf_embedded[:,1].max() - MAT_tfidf_embedded[:,1].min()),
            )
            # .set_index("TWEET_ID")
        ]
    )

DF_tfidf_embedded = (DF_tfidf_embedded.assign(LIST_2 = lambda df_: df_.LIST_2.apply(lambda x: "<br>".join([x[i:i+40] for i in range(0,len(x),40)]))))


# FIGURES

logging.info("Creating scatterplot chart.")

fig = (
    px.scatter(
        DF_tfidf_embedded,
        x = "x",
        y = "y",
        color = "CLUSTER_REAL",
        category_orders = {'CLUSTER_REAL': [str(i).zfill(2) for i in range(1,n_clusters+1)]},
        color_discrete_map = my_palette,
        opacity = 0.8,
        facet_col = "DR_model",
        custom_data = ["LIST_2", "CLUSTER_REAL"] + [f"top_word_{i}" for i in range(1,6)]
    )
    .update_layout(
        title = {
            "text": "<b>Tweets in embedded spaces by CLUSTER_REAL</b>",
            "x": 0.5
        },
        legend = {
            "y": 0.5
        }
    )
    .for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    .update_traces(
        hovertemplate = "<br>".join(
            [
                "(<b>%{x:7.5f}</b>, <b>%{y:7.5f}</b>)",
                "",
                "<b>Tweet:</b>",
                "%{customdata[0]}",
                "",
                "Most frequent words in cluster <b>%{customdata[1]}</b>:",
                "   - %{customdata[2]}",
                "   - %{customdata[3]}",
                "   - %{customdata[4]}",
                "   - %{customdata[5]}",
                "   - %{customdata[6]}"
            ]
        )
    )
)

# fig = px.scatter(
#     DF_tfidf_embedded,
#     x = "x",
#     y = "y",
#     color = "CLUSTER_REAL",
#     color_discrete_map = my_palette,
#     opacity = 0.8,
#     facet_col = "DR_model",
#     # size = pd.Series([1]*DF_tfidf_embedded.shape[0])
# )

# fig.update_layout(
#     title = "Tweets in embedded spaces by CLUSTER_REAL",
#     legend = dict(x = 0, y = -0.1, orientation = "h")
# )

fig.write_html(os.path.join(FIGURES_PATH, "2_scatterplot.html"))

# SAVE RESULTS

logging.info("Saving embedded data.")

DF_tfidf_embedded.to_csv(os.path.join(BBDD_PATH_3, "tfidf_embedded.csv.gz"), sep = delimiter, compression = "gzip")

logging.info("Dimenionality reduction finished.")
logging.info("-"*120)