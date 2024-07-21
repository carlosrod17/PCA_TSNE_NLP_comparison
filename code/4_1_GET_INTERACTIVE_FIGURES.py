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

from plotly.subplots import make_subplots

from sklearn.metrics import confusion_matrix

from var_def import BBDD_PATH_3
from var_def import BBDD_PATH_4
from var_def import FIGURES_PATH

from var_def import delimiter
from var_def import n_clusters

from var_def import my_palette
from var_def import clustering_colors

from functions import get_confusion_matrix

#####

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

scatterplot2 = (
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
        width = 1000,
        height = 400,
        margin = dict(t = 60, b = 60, l = 20, r = 20),
        title = dict(text = "<b>Tweets in embedded spaces by CLUSTER_REAL</b>", x = 0.5, font = dict(size = 12)),
        legend = dict(title_text = "", orientation = "h", yanchor = "bottom", xanchor = "center", x = 0.5, y = -0.3, font = dict(size = 10)),
        xaxis = dict(title = dict(text = "PCA", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis2 = dict(title = dict(text = "TSNE", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis3 = dict(title = dict(text = "PCA_TSNE", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis2 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis3 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),

    )
    .for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
)

for annotation in scatterplot2.layout.annotations:
    annotation['text'] = ''

scatterplot2.write_html(os.path.join(FIGURES_PATH, "2_scatterplot.html"))

#####

DF_silhouette1 = pd.read_csv(
    os.path.join(BBDD_PATH_4, "3_1_silhouette.csv"),
    sep = delimiter,
    header = 0,
    dtype = {
        "DR_model": str,
        "C_model": str,
        "s_score": float
    }
)

silhouette31 = (
    px.bar(
        DF_silhouette1,
        x = "DR_model",
        y = "s_score",
        color = "C_model",
        color_discrete_map = clustering_colors,
        barmode = "group",
    )
    .update_layout(
        width = 800,
        height = 600,
        margin = dict(t = 50, b = 80, l = 80, r = 50),
        title = dict(text = "<b>Silhouette coefficients for K-fixed classifications</b>", x = 0.5, font = dict(size = 12)),
        legend = dict(title_text = "", orientation = "h", yanchor = "bottom", xanchor = "center", x = 0.495, y = -0.2, font = dict(size = 10)),
        xaxis = dict(title = dict(text = "Dimensionality Reduction model", font = dict(size = 10)), tickfont = dict(size = 10)),
        yaxis = dict(title = dict(text = "Silhouette coefficient", font = dict(size = 10)), tickfont = dict(size = 7), range = [0,1])
    )
)

silhouette31.write_html(os.path.join(FIGURES_PATH, "3_1_silhouette.html"))


#####

DF_tfidf_classification_31 = (
        pd.read_csv(
        os.path.join(BBDD_PATH_4, "3_1_classification.csv.gz"),
        sep = delimiter,
        compression = "gzip",
        header = 0,
        dtype = "str"
    )
    .assign(
        x = lambda df_: df_.x.astype(float),
        y = lambda df_: df_.y.astype(float)
    )
)

#####

scatterplot31 = (
    px.scatter(
        DF_tfidf_classification_31,
        x = "x",
        y = "y",
        color = "CLUSTER",
        category_orders = {'CLUSTER': [str(i).zfill(2) for i in range(1,len(DF_tfidf_classification_31.CLUSTER.unique())+1)]},
        color_discrete_map = my_palette,
        opacity = 0.8,
        facet_col = "CLUSTER_TYPE",
        facet_row = "DR_model",
        facet_row_spacing = 0.05
    )
    .update_layout(
        width = 1000,
        height = 350*len(DF_tfidf_classification_31["DR_model"].unique())+100,
        margin = dict(t = 60, b = 60, l = 60, r = 90),
        title = dict(text = "<b>Real classifications vs. k-fixed classifications</b>", x = 0.5, font = dict(size = 12)),
        legend = dict(orientation = "v", yanchor = "middle", xanchor = "right", x = 1.08, y = 0.5, font = dict(size = 10)),
        xaxis = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis2 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis3 = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis4 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis5 = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis6 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis = dict(title = dict(text = "PCA_TSNE", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis2 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis3 = dict(title = dict(text = "TSNE", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis4 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis5 = dict(title = dict(text = "PCA", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis6 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),

    )
    .for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
)

for annotation in scatterplot31.layout.annotations:
    annotation['text'] = ''

scatterplot31.write_html(os.path.join(FIGURES_PATH, "3_1_scatterplot.html"))


#####

y_true = (
    DF_tfidf_classification_31
    .pipe(lambda df_: df_[df_.CLUSTER_TYPE == "Real"])
    .pipe(lambda df_: df_[df_.DR_model == "PCA_TSNE"])
    .filter(["TWEET_ID", "CLUSTER"])
    .set_index("TWEET_ID")
)
y_pred = [
    (
        DF_tfidf_classification_31
        .pipe(lambda df_: df_[df_.CLUSTER_TYPE == "Estimated"])
        .pipe(lambda df_: df_[df_.DR_model == dr_model])
        .filter(["TWEET_ID", "CLUSTER"])
        .set_index("TWEET_ID")
    ) 
    for dr_model in ["PCA", "TSNE", "PCA_TSNE"]
]

pred_labels = [np.unique(y_pred[i]).shape[0] for i in range(len(y_pred))]
width_ratios = [i/sum(pred_labels) for i in pred_labels]

confusion31 = make_subplots(
    rows = 1,
    cols = len(y_pred),
    column_widths = width_ratios,
    horizontal_spacing = 0.04
)

for i in range(len(y_pred)):

    labels = list(set(np.unique(y_true)) | set(np.unique(y_pred[i])))

    DF_CM = (
        pd.DataFrame(
            data = confusion_matrix(
                y_true = y_true.sort_index(axis = 0).CLUSTER.to_list(),
                y_pred = y_pred[i].sort_index(axis = 0).CLUSTER.to_list(),
                labels = labels
            ),
            columns = labels,
            index = labels
        )
        .loc[
            list(set(np.unique(y_true))),
            list(set(np.unique(y_pred[i])))
        ]
    )

    DF_CM = DF_CM.sort_index(axis = 0, ascending = False).sort_index(axis = 1)

    heatmap = (
        px.imshow(
            DF_CM,
        )
    )

    for trace in heatmap.data:
        confusion31.add_trace(
            trace,
            row = 1, 
            col = i+1
        )

confusion31.update_layout(
    width = 1000,
    height = 450,
    margin = dict(t = 60, b = 60, l = 20, r = 20),
    title = dict(text = "<b>K-fixed classifications' Confusion Matrices</b>", x = 0.5, font = dict(size = 12)),
    coloraxis=dict(
        colorscale='Viridis', 
        colorbar=dict(title = "", orientation='h', xanchor='center', x=0.5, y=-0.35, len = 0.6, thickness = 10, tickvals = [0,250,500,750,1000], tickfont = dict(size = 10))
    ),
    xaxis = dict(title = dict(text = "PCA", font = dict(size = 10)), tickfont=dict(size = 8)),
    xaxis2 = dict(title = dict(text = "TSNE", font = dict(size = 10)), tickfont=dict(size = 8)),
    xaxis3 = dict(title = dict(text = "PCA_TSNE", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis2 = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis3 = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
)

confusion31.write_html(os.path.join(FIGURES_PATH, "3_1_confusionmatrix.html"))


######

DF_silhouette2 = pd.read_csv(
    os.path.join(BBDD_PATH_4, "3_2_silhouette.csv"),
    sep = delimiter,
    header = 0,
    dtype = {
        "DR_model": str,
        "C_model": str,
        "n_clusters": int, 
        "s_score": float
    }
)

silhouette32 = (
    px.line(
        DF_silhouette2,
        x = "n_clusters",
        y = "s_score",
        color = "C_model",
        color_discrete_map = clustering_colors,
        facet_row = "DR_model"
    )
    .update_layout(
        width = 900,
        height = 600,
        margin = dict(t = 50, b = 80, l = 80, r = 50),
        title = dict(text = "<b>Silhouette coefficients for K-optimized classifications</b>", x = 0.5, font = dict(size = 12)),
        legend = dict(title_text = "", orientation = "h", yanchor = "bottom", xanchor = "center", x = 0.495, y = -0.17, font = dict(size = 10)),
        xaxis = dict(title = dict(text = "n_clusters", font = dict(size = 10)), tickfont = dict(size = 10)),
        yaxis = dict(title = dict(text = "PCA_TSNE", font = dict(size = 10)), tickvals = [0, 0.25, 0.5, 0.75, 1], tickfont = dict(size = 7), range = [0,1]),
        yaxis2 = dict(title = dict(text = "TSNE", font = dict(size = 10)), tickvals = [0, 0.25, 0.5, 0.75, 1], tickfont = dict(size = 7), range = [0,1]),
        yaxis3 = dict(title = dict(text = "PCA", font = dict(size = 10)), tickvals = [0, 0.25, 0.5, 0.75, 1], tickfont = dict(size = 7), range = [0,1]),
    )
)

for annotation in silhouette32.layout.annotations:
    annotation['text'] = ''

for row_index, facet_value in enumerate(DF_silhouette2.DR_model.unique()):

    max_score_id = (
        DF_silhouette2
        .pipe(lambda df_: df_[df_.DR_model == facet_value])
        ["s_score"].idxmax()
    )

    silhouette32.add_scatter(
        x = [DF_silhouette2.loc[max_score_id, "n_clusters"]],
        y = [DF_silhouette2.loc[max_score_id, "s_score"]],
        marker=dict(color = clustering_colors[DF_silhouette2.loc[max_score_id, "C_model"]], size = 10),
        name=None,
        showlegend=False,
        row = len(DF_silhouette2.DR_model.unique()) - row_index,
        col = 1
    )

silhouette32.write_html(os.path.join(FIGURES_PATH, "3_2_silhouette.html"))


#####

DF_tfidf_classification_32 = (
        pd.read_csv(
        os.path.join(BBDD_PATH_4, "3_2_classification.csv.gz"),
        sep = delimiter,
        compression = "gzip",
        header = 0,
        dtype = "str"
    )
    .assign(
        x = lambda df_: df_.x.astype(float),
        y = lambda df_: df_.y.astype(float)
    )
)

#####

scatterplot32 = (
    px.scatter(
        DF_tfidf_classification_32,
        x = "x",
        y = "y",
        color = "CLUSTER",
        category_orders = {'CLUSTER': [str(i).zfill(2) for i in range(1,len(DF_tfidf_classification_32.CLUSTER.unique())+1)]},
        color_discrete_map = my_palette,
        opacity = 0.8,
        facet_col = "CLUSTER_TYPE",
        facet_row = "DR_model",
        facet_row_spacing = 0.05
    )
    .update_layout(
        width = 1000,
        height = 350*len(DF_tfidf_classification_32["DR_model"].unique())+100,
        margin = dict(t = 60, b = 60, l = 60, r = 90),
        title = dict(text = "<b>Real classifications vs. k-optimized classifications</b>", x = 0.5, font = dict(size = 12)),
        legend = dict(orientation = "v", yanchor = "middle", xanchor = "right", x = 1.08, y = 0.5, font = dict(size = 10)),
        xaxis = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis2 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis3 = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis4 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis5 = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis6 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis = dict(title = dict(text = "PCA_TSNE", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis2 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis3 = dict(title = dict(text = "TSNE", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis4 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis5 = dict(title = dict(text = "PCA", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis6 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),

    )
    .for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
)

for annotation in scatterplot32.layout.annotations:
    annotation['text'] = ''

scatterplot32.write_html(os.path.join(FIGURES_PATH, "3_2_scatterplot.html"))

#####

y_true = (
    DF_tfidf_classification_32
    .pipe(lambda df_: df_[df_.CLUSTER_TYPE == "Real"])
    .pipe(lambda df_: df_[df_.DR_model == "PCA"])
    .filter(["TWEET_ID", "CLUSTER"])
    .set_index("TWEET_ID")
)
y_pred = [
    (
        DF_tfidf_classification_32
        .pipe(lambda df_: df_[df_.CLUSTER_TYPE == "Estimated"])
        .pipe(lambda df_: df_[df_.DR_model == dr_model])
        .filter(["TWEET_ID", "CLUSTER"])
        .set_index("TWEET_ID")
    ) 
    for dr_model in ["PCA", "TSNE", "PCA_TSNE"]
]

pred_labels = [np.unique(y_pred[i]).shape[0] for i in range(len(y_pred))]
width_ratios = [i/sum(pred_labels) for i in pred_labels]

confusion32 = make_subplots(
    rows = 1,
    cols = len(y_pred),
    column_widths = width_ratios,
    horizontal_spacing = 0.04
)

for i in range(len(y_pred)):

    labels = list(set(np.unique(y_true)) | set(np.unique(y_pred[i])))

    DF_CM = (
        pd.DataFrame(
            data = confusion_matrix(
                y_true = y_true.sort_index(axis = 0).CLUSTER.to_list(),
                y_pred = y_pred[i].sort_index(axis = 0).CLUSTER.to_list(),
                labels = labels
            ),
            columns = labels,
            index = labels
        )
        .loc[
            list(set(np.unique(y_true))),
            list(set(np.unique(y_pred[i])))
        ]
    )


    DF_CM = DF_CM.sort_index(axis = 0, ascending = False).sort_index(axis = 1)

    heatmap = (
        px.imshow(
            DF_CM,
            color_continuous_scale = 'rdbu',
            zmin = 0,
            zmax = 1000
        )
    )

    for trace in heatmap.data:
        confusion32.add_trace(
            trace,
            row = 1, 
            col = i+1
        )

confusion32.update_layout(
    width = 1000,
    height = 450,
    margin = dict(t = 60, b = 60, l = 20, r = 20),
    title = dict(text = "<b>K-optimized classifications' Confusion Matrices</b>", x = 0.5, font = dict(size = 12)),
    coloraxis=dict(
        colorscale='Viridis', 
        colorbar=dict(title = "", orientation='h', xanchor='center', x=0.5, y=-0.35, len = 0.6, thickness = 10, tickvals = [0,250,500,750,1000], tickfont = dict(size = 10))
    ),
    xaxis = dict(title = dict(text = "PCA", font = dict(size = 10)), tickfont=dict(size = 8)),
    xaxis2 = dict(title = dict(text = "TSNE", font = dict(size = 10)), tickfont=dict(size = 8)),
    xaxis3 = dict(title = dict(text = "PCA_TSNE", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis2 = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis3 = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
)

confusion32.write_html(os.path.join(FIGURES_PATH, "3_2_confusionmatrix.html"))


#####

DF_tfidf_classification_33 = (
        pd.read_csv(
        os.path.join(BBDD_PATH_4, "3_3_classification.csv.gz"),
        sep = delimiter,
        compression = "gzip",
        header = 0,
        dtype = "str"
    )
    .assign(
        x = lambda df_: df_.x.astype(float),
        y = lambda df_: df_.y.astype(float)
    )
)

#####

scatterplot33 = (
    px.scatter(
        DF_tfidf_classification_33,
        x = "x",
        y = "y",
        color = "CLUSTER",
        category_orders = {'CLUSTER': [str(i).zfill(2) for i in range(1,len(DF_tfidf_classification_33.CLUSTER.unique())+1)]},
        color_discrete_map = my_palette,
        opacity = 0.8,
        facet_col = "CLUSTER_TYPE",
        facet_row = "DR_model",
        facet_row_spacing = 0.05
    )
    .update_layout(
        width = 1000,
        height = 350*len(DF_tfidf_classification_33["DR_model"].unique())+100,
        margin = dict(t = 60, b = 60, l = 60, r = 90),
        title = dict(text = "<b>Real classifications vs. DBSCAN classifications</b>", x = 0.5, font = dict(size = 12)),
        legend = dict(orientation = "v", yanchor = "middle", xanchor = "right", x = 1.08, y = 0.5, font = dict(size = 10)),
        xaxis = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis2 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis3 = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis4 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis5 = dict(title = dict(text = "Real", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        xaxis6 = dict(title = dict(text = "K-MEANS", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis = dict(title = dict(text = "PCA_TSNE", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis2 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis3 = dict(title = dict(text = "TSNE", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis4 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis5 = dict(title = dict(text = "PCA", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),
        yaxis6 = dict(title = dict(text = "", font = dict(size = 10)), tickvals = [0, 0.5, 1], showticklabels=False),

    )
    .for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
)

for annotation in scatterplot33.layout.annotations:
    annotation['text'] = ''

scatterplot33.write_html(os.path.join(FIGURES_PATH, "3_3_scatterplot.html"))


#####

y_true = (
    DF_tfidf_classification_33
    .pipe(lambda df_: df_[df_.CLUSTER_TYPE == "Real"])
    .pipe(lambda df_: df_[df_.DR_model == "PCA"])
    .filter(["TWEET_ID", "CLUSTER"])
    .set_index("TWEET_ID")
)

y_pred = [
    (
        DF_tfidf_classification_33
        .pipe(lambda df_: df_[df_.CLUSTER_TYPE == "Estimated"])
        .pipe(lambda df_: df_[df_.DR_model == dr_model])
        .filter(["TWEET_ID", "CLUSTER"])
        .set_index("TWEET_ID")
    ) 
    for dr_model in ["PCA", "TSNE", "PCA_TSNE"]
]

pred_labels = [np.unique(y_pred[i]).shape[0] for i in range(len(y_pred))]
width_ratios = [i/sum(pred_labels) for i in pred_labels]

confusion33 = make_subplots(
    rows = 1,
    cols = len(y_pred),
    column_widths = width_ratios,
    horizontal_spacing = 0.04
)

for i in range(len(y_pred)):

    labels = list(set(np.unique(y_true)) | set(np.unique(y_pred[i])))

    DF_CM = (
        pd.DataFrame(
            data = confusion_matrix(
                y_true = y_true.sort_index(axis = 0).CLUSTER.to_list(),
                y_pred = y_pred[i].sort_index(axis = 0).CLUSTER.to_list(),
                labels = labels
            ),
            columns = labels,
            index = labels
        )
        .loc[
            list(set(np.unique(y_true))),
            list(set(np.unique(y_pred[i])))
        ]
    )


    DF_CM = DF_CM.sort_index(axis = 0, ascending = False).sort_index(axis = 1)

    heatmap = (
        px.imshow(
            DF_CM,
            color_continuous_scale = 'rdbu',
            zmin = 0,
            zmax = 1000
        )
    )

    for trace in heatmap.data:
        confusion33.add_trace(
            trace,
            row = 1, 
            col = i+1
        )

confusion33.update_layout(
    width = 1000,
    height = 450,
    margin = dict(t = 60, b = 60, l = 20, r = 20),
    title = dict(text = "<b>DBSCAN classifications' Confusion Matrices</b>", x = 0.5, font = dict(size = 12)),
    coloraxis=dict(
        colorscale='Viridis', 
        colorbar=dict(title = "", orientation='h', xanchor='center', x=0.5, y=-0.35, len = 0.6, thickness = 10, tickvals = [0,250,500,750,1000], tickfont = dict(size = 10))
    ),
    xaxis = dict(title = dict(text = "PCA", font = dict(size = 10)), tickfont=dict(size = 8)),
    xaxis2 = dict(title = dict(text = "TSNE", font = dict(size = 10)), tickfont=dict(size = 8)),
    xaxis3 = dict(title = dict(text = "PCA_TSNE", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis2 = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
    yaxis3 = dict(title = dict(text = "", font = dict(size = 10)), tickfont=dict(size = 8)),
)

confusion33.write_html(os.path.join(FIGURES_PATH, "3_3_confusionmatrix.html"))