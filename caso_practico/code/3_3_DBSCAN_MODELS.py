##############################################################################
# DBSCAN CLASSIFICATION
##############################################################################

import os
import sys
import logging
import datetime
import pytz
import numpy as np
import pandas as pd
import plotly.express as px  

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from var_def import TFM_PATH
from var_def import BBDD_PATH_2
from var_def import BBDD_PATH_3
from var_def import BBDD_PATH_4
from var_def import FIGURES_PATH

from var_def import delimiter

from var_def import DR_models_to_perform

from var_def import my_palette

from functions import get_optim_cluster

logging.basicConfig(
    filename = os.path.join(TFM_PATH, sys.argv[1]),
    format="%(asctime)s %(levelname)s: %(message)s",
    level = logging.INFO)

logging.info("-"*120)
logging.info(f"Starting DBSCAN clustering.")

n_DR = len(DR_models_to_perform)

logging.info("Loading tfidfs embedded.")

DF_tfidf_embedded = pd.read_csv(
    os.path.join(BBDD_PATH_3, "tfidf_embedded.csv.gz"),
    sep = delimiter,
    compression = "gzip",
    header = 0,
    dtype = {
        "TWEET_ID": str,
        "CLUSTER_REAL": str,
        "DR_model": str,
        "x": float,
        "y": float
    }
)

logging.info(f"{len(DF_tfidf_embedded.index.unique()):06d} tfidfs embedded have being loaded.")

# PREPARATION

logging.info("Prepararing dataframes to store the results.")

DF_metrics = pd.DataFrame(
    data = 0,
    columns = DR_models_to_perform,
    index = ['C_model', "eps", "min_samples", "n_cluster", 's_score', 'a_score', 'p_scores']
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
    
    eps_vect = np.linspace(
        0,
        0.15*max(np.max(abs(data["x"])), np.max(abs(data["y"]))),
        21
    )
    
    eps_vect = np.delete(eps_vect, 0)
    
    min_samples_vect = np.arange(2,26)
    
    for eps in eps_vect:
        
        for min_samples in min_samples_vect:

            logging.info(f"   Performing DBSCAN classification with parameters: eps = {eps:7.5f}, min_samples = {min_samples:02d}")
    
            dbscan = DBSCAN(eps = eps,
                            min_samples = int(min_samples),
                            n_jobs = -1)
            
            dbscan.fit(data)

            k = np.unique(dbscan.labels_ != -1).shape[0]
            
            if k < 2:
                logging.info(f"      + The DBSCAN classification just obtained one cluster. Model discarded.")
                continue

            s_score = silhouette_score(data, dbscan.labels_)

            logging.info(f"      + The DBSCAN classification obtained {s_score:10.8f} of silhouette coeficient.")

            if s_score > DF_metrics.loc["s_score", dr_model]:
                DF_metrics.loc['n_clusters', dr_model] = k
                DF_metrics.loc['eps', dr_model] = eps
                DF_metrics.loc['min_samples', dr_model] = min_samples
                DF_metrics.loc['s_score',dr_model] = s_score

    logging.info(f"   The best classification for {dr_model}-embedded data is DBSCAN with parameters:")
    logging.info(f"      - eps = {DF_metrics.loc['eps', dr_model]:7.5f}")
    logging.info(f"      - min_samples = {int(DF_metrics.loc['min_samples', dr_model]):02d}")
                                      
    dbscan = DBSCAN(eps = DF_metrics.loc['eps',dr_model],
                    min_samples = int(DF_metrics.loc['min_samples',dr_model]),
                    n_jobs = -1)
            
    dbscan.fit(data)

    logging.info("   Rearranging labels to maximize the accuracy.")
    
    not_outliers = dbscan.labels_ != -1
    outliers = dbscan.labels_ == -1
    
    OPTIM_CLUSTER = dbscan.labels_
        
    OPTIM_CLUSTER[not_outliers] = get_optim_cluster(TRUE_CLUSTER[not_outliers]-1, dbscan.labels_[not_outliers])+1
    OPTIM_CLUSTER[outliers] = 0    
    
    k = np.unique(OPTIM_CLUSTER != 0).shape[0]
    s_score = silhouette_score(data, OPTIM_CLUSTER)                     
    a_score = accuracy_score(y_true = TRUE_CLUSTER,
                             y_pred = OPTIM_CLUSTER)
    p_score = precision_score(y_true = TRUE_CLUSTER,
                              y_pred = OPTIM_CLUSTER,
                              average = None, 
                              labels = sorted(list(np.unique(OPTIM_CLUSTER))))
    
    logging.info(f"      + The {dr_model}-embedded data classification obtained {k} components.")
    logging.info(f"      + The {dr_model}-embedded data classification obtained {s_score} of silhouette coeficient.")
    logging.info(f"      + The {dr_model}-embedded data classification obtained {a_score} of accuracy.")
                       
    DF_metrics.loc['n_clusters', dr_model] = k
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

# FIGURES

# SCATTER PLOT BY ERROR AND CONFUSION MATRIX

logging.info("Creating scatterplot chart.")

fig = (
    px.scatter(
        DF_classification,
        x = "x",
        y = "y",
        color = "CLUSTER",
        category_orders={'CLUSTER': [str(i).zfill(2) for i in range(1,len(DF_classification.CLUSTER.unique())+1)]},
        color_discrete_map = my_palette,
        opacity = 0.8,
        facet_row = "CLUSTER_TYPE",
        facet_col = "DR_model",
        custom_data = ["LIST_2", "CLUSTER"] + [f"top_word_{i}" for i in range(1,6)]
        # hover_data = {"DR_model": False, "CLUSTER_TYPE": False, "CLUSTER": True, "TOP_WORDS": True},
    )
    .update_layout(
        title = {
            "text": "<b>Real classifications vs. DBSCAN classifications</b>",
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

fig.write_html(os.path.join(FIGURES_PATH, "3_3_scatterplot.html"))

# get_confusion_matrix(dim_red_names = [dim_red_names[i] for i in dim_red_models],
#                      y_true = DF_classification['CLUSTER_REAL'],
#                      y_pred = [DF_classification[dim_red_abrs[i]] for i in dim_red_models],
#                      vmin = vmin,
#                      vmax = vmax,
#                      w = 3.3,
#                      suptitle = SUPTITLE_KFREE_DBSCAN_CM,
#                      colormap = 'coolwarm',
#                      path = FIGURES_PATH + FILENAME_KFREE_DBSCAN_CM)


# SAVE RESULTS

logging.info("Saving results.")

DF_metrics.to_csv(os.path.join(BBDD_PATH_4, "3_3_metrics.csv"), sep = delimiter)
DF_classification.to_csv(os.path.join(BBDD_PATH_4, "3_3_classification.csv.gz"), sep = delimiter, compression = "gzip")

logging.info(f"DBSCAN clustering finished.")
logging.info("-"*120)