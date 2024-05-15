##############################################################################
# CLASSIFICATION (K FIXED)
##############################################################################

import os
import sys
import logging
import datetime
import pytz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px   

from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from var_def import TFM_PATH
from var_def import BBDD_PATH_2
from var_def import BBDD_PATH_3
from var_def import BBDD_PATH_4
from var_def import FIGURES_PATH

from var_def import delimiter
from var_def import n_clusters

from var_def import DR_models_to_perform
from var_def import C_models_to_perform
from var_def import C_models

from var_def import my_palette

from functions import fit_predict_model
from functions import get_optim_cluster

logging.basicConfig(
    filename = os.path.join(TFM_PATH, sys.argv[1]),
    format="%(asctime)s %(levelname)s: %(message)s",
    level = logging.INFO)

logging.info("-"*120)
logging.info(f"Starting clustering with k = {n_clusters} fixed.")

n_DR = len(DR_models_to_perform)

logging.info("Loading tfidfs embedded.")

DF_tfidf_embedded = pd.read_csv(
    os.path.join(BBDD_PATH_3, "tfidf_embedded.csv.gz"),
    sep = delimiter,
    compression = "gzip",
    header = 0,
    dtype = {
        "TWEET_ID": str,
        "CLUSTER": str,
        "CLUSTER_REAL": str,
        "DR_model": str,
        "x": float,
        "y": float
    }
)

logging.info(f"{len(DF_tfidf_embedded.index.unique()):06d} tfidfs embedded have being loaded.")

# PREPARATION

logging.info("Prepararing dataframes to store the results.")

DF_silhouette = pd.DataFrame(
    data = None,
    columns = ["DR_model", "C_model", "s_score"]
)

DF_metrics = pd.DataFrame(
    data = 0,
    columns = DR_models_to_perform,
    index = ['C_model', 's_score', 'a_score', 'p_scores']
)

DF_classification = (
    DF_tfidf_embedded
    # .rename(columns = {"CLUSTER_REAL": "CLUSTER"})
    .assign(CLUSTER = lambda df_: df_.CLUSTER_REAL.astype(str).str.zfill(2))
    .assign(CLUSTER_TYPE = "Real")
    .set_index("TWEET_ID")
)

logging.info("Dataframes prepared.")

# IMPLEMENTATION

for dr_model in DR_models_to_perform:
    
    logging.info(f"Classificating {dr_model}-embedded data.")

    DF_aux = (
        DF_tfidf_embedded
        .pipe(lambda df_: df_[df_.DR_model == dr_model])
        .pipe(lambda df_: df_[["TWEET_ID", "CLUSTER", "LIST_2", "DR_model", "x", "y"]])
        .assign(CLUSTER_TYPE = "Estimated")
        .set_index("TWEET_ID")
    )

    TRUE_CLUSTER = DF_aux["CLUSTER"].astype(int)
    
    data = (
        DF_aux
        .filter(["x", "y"])
    )
    
    for c_model in C_models_to_perform:
        
        logging.info(f"   Performing {c_model} classification.")
        
        labels = fit_predict_model(data = data,
                                   model = c_model,
                                   params = C_models[c_model]["parameters"],
                                   verbose = True)
        
        s_score = silhouette_score(data, labels)
        
        logging.info(f"         + The {c_model} classification obtained {s_score} of silhouette coeficient.")
        
        DF_silhouette = pd.concat(
            [
                DF_silhouette,
                pd.DataFrame({"DR_model": [dr_model], "C_model": [c_model], "s_score": [s_score]})
            ]
        )
        
        if s_score > DF_metrics.loc["s_score", dr_model]:
            DF_metrics.loc['C_model', dr_model] = c_model
            DF_metrics.loc['s_score', dr_model] = s_score
    
    best_model = DF_metrics.loc['C_model', dr_model]
    
    logging.info(f"   The best classification for {dr_model}-embedded data is {best_model}.")
    
    labels = fit_predict_model(data = data,
                               model = DF_metrics.loc['C_model', dr_model],
                               params = C_models[best_model]["parameters"],
                               verbose = True)
    
    logging.info("   Rearranging labels to maximize the accuracy.")
    
    OPTIM_CLUSTER = get_optim_cluster(TRUE_CLUSTER-1, labels)+1
      
    s_score = silhouette_score(data, OPTIM_CLUSTER)                     
    a_score = accuracy_score(y_true = TRUE_CLUSTER,
                             y_pred = OPTIM_CLUSTER)
    p_score = precision_score(y_true = TRUE_CLUSTER,
                              y_pred = OPTIM_CLUSTER,
                              average = None, 
                              labels = sorted(list(np.unique(OPTIM_CLUSTER))))
    
    logging.info(f"      + The {dr_model}-embedded data classification obtained {s_score} of silhouette coeficient.")
    logging.info(f"      + The {dr_model}-embedded data classification obtained {a_score} of accuracy.")
    
    DF_metrics.loc['s_score', dr_model] = s_score
    DF_metrics.loc['a_score', dr_model] = 100*a_score
    DF_metrics.loc['p_scores', dr_model] = " & ".join([f"{100*elem:.1f}" for elem in list(p_score)])

    DF_top_words = pd.DataFrame(
        pd.merge(
            pd.read_csv(
                os.path.join(BBDD_PATH_2, "tweets_tfidf.csv.gz"),
                sep = delimiter,
                compression = "gzip",
                header = 0,
                index_col = "TWEET_ID"
            )
            .astype(float),
            DF_aux
            .assign(CLUSTER = OPTIM_CLUSTER)
            .assign(CLUSTER = lambda df_: df_.CLUSTER.astype(str).str.zfill(2))
            .filter(["CLUSTER"]),
            on = "TWEET_ID"
        )
        .groupby("CLUSTER").sum()
        .apply(lambda row: pd.Series(row.nlargest(5).index.to_list()), axis = 1)
    .reset_index()
    .rename(columns = {i: f"top_word_{i+1}" for i in range(5)})
    )

    DF_classification = pd.concat(
        [
            DF_classification,
            pd.merge(
                DF_top_words,
                DF_aux
                .assign(CLUSTER = OPTIM_CLUSTER)
                .assign(CLUSTER = lambda df_: df_.CLUSTER.astype(str).str.zfill(2))
                .reset_index(),
                on = "CLUSTER"
            )
            .set_index("TWEET_ID")
        ]
    )

# SILHOUETTES BARS

logging.info("Creating barplot chart of silhouette coeficients.")

fig = px.bar(
    DF_silhouette,
    x = "DR_model",
    y = "s_score",
    color = "C_model",
    barmode = "group"
)

fig.update_layout(
    title = "Silhouette coefficients for K-fixed classifications",
    xaxis_title = "Dimensionality Reduction model",
    yaxis_title = "Silhouette coefficient",
    yaxis_range = [0,1]
)

fig.write_html(os.path.join(FIGURES_PATH, "3_1_silhouette.html"))

# FIGURES

# SCATTER PLOT BY ERROR AND CONFUSION MATRIX

logging.info("Creating scatterplot chart.")

fig = (
    px.scatter(
        DF_classification,
        x = "x",
        y = "y",
        color = "CLUSTER",
        category_orders = {'CLUSTER': [str(i).zfill(2) for i in range(1,len(DF_classification.CLUSTER.unique())+1)]},
        color_discrete_map = my_palette,
        opacity = 0.8,
        facet_row = "CLUSTER_TYPE",
        facet_col = "DR_model",
        custom_data = ["LIST_2", "CLUSTER"] + [f"top_word_{i}" for i in range(1,6)]
        # hover_data = {"DR_model": False, "CLUSTER_TYPE": False, "CLUSTER": True, "TOP_WORDS": True},
    )
    .update_layout(
        title = {
            "text": "<b>Real classifications vs. k-fixed classifications</b>",
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

fig.write_html(os.path.join(FIGURES_PATH, "3_1_scatterplot.html"))

"""

logging.info(f"Creating confusion matrices chart.")

get_confusion_matrix(dim_red_names = [DR_models[dr_model]["short_name"] for dr_model in DR_models_to_perform],
                     y_true = DF_classification['CLUSTER_REAL'],
                     y_pred = [DF_classification[dr_model] for dr_model in DR_models_to_perform],
                     vmin = vmin,
                     vmax = vmax,
                     w = 6.05,
                     suptitle = "K-fixed classifications' Confusion Matrices",
                     colormap = 'coolwarm',
                     path = os.path.join(FIGURES_PATH, "3_1_confusionmatrix.png"))

"""


# SAVE RESULTS

logging.info("Saving results.")

DF_silhouette.to_csv(os.path.join(BBDD_PATH_4, "3_1_silhouette.csv"), sep = delimiter)
DF_metrics.to_csv(os.path.join(BBDD_PATH_4, "3_1_metrics.csv"), sep = delimiter)
DF_classification.to_csv(os.path.join(BBDD_PATH_4, "3_1_classification.csv.gz"), sep = delimiter, compression = "gzip")

logging.info(f"Clustering with k = {n_clusters} fixed finished.")
logging.info("-"*120)