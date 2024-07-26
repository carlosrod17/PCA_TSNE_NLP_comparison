import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

from mip import Model, xsum, maximize

def fit_predict_model(data, model, params, verbose):
    
    """
    
    fit_predict_model implements a clustering model (selected from a set of
    available models) on a dataset, classificating its point into k clusters.
    The model's training parameters are set and the seed is fixed to ensure
    replicability.
    
    Input:
        - model (str): should be one of the following: 'KMEANS', 'AGLO',
          'GMIXT'. If not, a warning will be printed.
        
        - data (NumPy Array or Pandas DataFrame): dataset that will clustered
          usign the selected model.
          
         - k (int): represent the number of clusters. k hould be greater
           than 2. 
          
    Output:
        - cluster_perdicted (NumPy Array): array of dimensions (n_samples,1),
          where n_samples is the number of points to be classified. The values
          of cluster_predicted range from 0 to 10 (inclusive) and represent
          the cluster to which each point has been assigned.
    
    
    """
    
    if model == 'KMEANS':

        if verbose:
            logging.info(f"      Building {model} classification model with parameters:")
            [logging.info(f"         - {key} = {value}") for key,value in params.items()]
        
        kmeans = KMeans(n_clusters = params["n_clusters"],
                        init = params["init"],
                        n_init = params["n_init"],
                        random_state = params["random_state"])
        
        if verbose:
            logging.info(f"      Classificating data with {model} model.")
        
        cluster_predicted = kmeans.fit_predict(data)
        
        if verbose:
            logging.info("      Data classificated succesfully.")
        
        return cluster_predicted
        
    elif model == 'AGLO':
        
        if verbose:
            logging.info(f"      Building {model} classification model with parameters:")
            [logging.info(f"         - {key} = {value}") for key,value in params.items()]
        
        aclustering = AgglomerativeClustering(n_clusters = params["n_clusters"],
                                              linkage = params["linkage"])
        
        if verbose:
            logging.info(f"      Classificating data with {model} model.")
        
        np.random.seed(params["random_state"])
        cluster_predicted = aclustering.fit_predict(data)
        
        if verbose:
            logging.info("      Data classificated succesfully.")
        
        return cluster_predicted
        
    elif model == 'GMIXT':
        
        if verbose:
            logging.info(f"      Building {model} classification model with parameters:")
            [logging.info(f"         - {key} = {value}") for key,value in params.items()]
        
        gmixture = GaussianMixture(n_components = params["n_clusters"],
                                   covariance_type = params["covariance_type"],
                                   init_params = params["init_params"],
                                   max_iter = params["max_iter"],
                                   random_state = params["random_state"])
        
        if verbose:
            logging.info(f"      Classificating data with {model} model.")
        
        cluster_predicted = gmixture.fit_predict(data)
        
        if verbose:
            logging.info("      Data classificated succesfully.")
        
        return cluster_predicted
    
    else:
        
        print('Something went wrong. The input model is not available.')    


def get_optim_cluster(y_true, y_pred):
    
    """
    
    get_optim_cluster takes two vectors of the same length that represent the
    true labels and the predicted labels of a classified dataset and computes
    a permutation of the predicted labels that maximizes the accuracy of the
    predicted classification. The two sets of unique labels may not be the same.
    
    Input:
        - y_true (Numpy Array or Pandas Series): Represents the true labels
          of a dataset.
        
        - y_pred (Numpy Array or Pandas Series): Represents the predicted
         labels of a model on the same dataset as y_true labels. It should
         have the same length as y_true.
    
    Output:
        - y_pred_final (Numpy Array or Pandas Series): An array of the same
          length as y_true and y_pred that represents the new predicted
          labels of the dataset, maximizing the accuracy of the classification.
        
        
    """
    
    # IMPLEMENTATION OF A MIP PROBLEM
    
    # DATA
    
    CM_INI = confusion_matrix(y_true = y_true,
                              y_pred = y_pred)
    
    # MODEL'S STRUCTURE

    model = Model("Model")
    
    # MODEL'S VARIABLES
    
    x = model.add_var_tensor(CM_INI.shape,
                             "x",
                             var_type = "B")
    
    # MODEL'S SETS OF INDICES

    I = range(CM_INI.shape[0])
    J = range(CM_INI.shape[1])
    
    # MODEL'S OBJECTIVE FUNCTION

    model.objective = maximize(xsum(xsum(CM_INI[i,j]*x[i,j] for j in J) for i in I))
    
    # MODEL'S CONSTRAINTS
    
    for i in I:
        model.add_constr(xsum(x[i,j] for j in J) == 1)
        
    for j in J:
        model.add_constr(xsum(x[i,j] for i in I) == 1)
        
    # RESOLUTION
    
    model.verbose = 0
    status = model.optimize()
    
    # GETTING THE PERMUTATION FROM THE SOLUTION
    
    PERM = np.zeros(CM_INI.shape[0], dtype = np.int8)
    
    for j in J:
        for i in I:
            if x[i,j].x > 0:
                PERM[j] = int(i)
    
    y_pred_final = PERM[y_pred]

    return y_pred_final


def get_scatter_plot(dim_red_names, clustering_names, x, y,
                     hue_true, hue_pred, suptitle, color_palette, path):

    """
    
    get_scatter_plot creates and saves a figure of scatterplot graphics for
    various datasets with two variables. For each dataset, there will be two
    scatterplot graphics: one representing the true classification and the
    other representing the predicted classification. The figure will have 
    two columns: the first column for the scatterplots of the true
    classifications and the second column for the scatterplots of the
    predicted classifications. The figure will have as many rows as datasets
    to represent. The datasets have the same number of points because they are
    different representations of the same original dataset.
    
    Input:
        - dim_red_names (list of str): A list of the names of the datasets.
          The elements will be the titles of the scatterplots in the first
          column of the figure.
          
        - clustering_names (list of str): A list of the names of the algorithms
          used to obtain the predicted classifications of the datasets. The
          titles of the scatterplots in the second column of the figure will
          have the names of the datasets and the names of these algorithms.
          It should have the same length as dim_red_names.
          
        - x (list of Numpy Arrays or Pandas Series): its elements represent the
          first coordinate of the points for each dataset. It should have the
          same length as dim_red_names, and its elements should have the same
          length.
          
        - y (list of Numpy Arrays or Pandas Series): its elements represent the
          second coordinate of the points for each dataset. It should have the
          same length as dim_red_names, and its elements should have the same
          length as the elements of x.
          
        - hue_true (Numpy Array or Pandas Series): Represents the labels of
          the true classification for the datasets (the same for all). It
          should have its same length as the elements of x and y.
          
        - hue_pred (list of Numpy Arrays or Pandas Series): its elements
          represent the labels of the predicted classification for the
          datasets. It should have the same length as dim_red_names, and its
          elements should have the same length as hue_true and the elements
          of x and y.
          
        - suptitle (str): The supertitle of the figure.
        
        - color_palette (list of RGB triples): A list of colors in RGB format
          in which the points will be represented.
          
        - path (str): The path, including the filename and extension, where
          the figure will be saved.
          
          
    """    
    
    n_DR = len(dim_red_names)
    
    fig, axes = plt.subplots(n_DR, 2, figsize = (6,2*n_DR+1))
    
    plt.subplots_adjust(bottom = 0.03,
                        top = 0.93,
                        left = 0.05,
                        right = 0.94,
                        wspace = 0.31,
                        hspace = 0.29)
    
    if n_DR == 1:
        axes = [[axes[0], axes[1]]]
           
    fig.suptitle(suptitle,
                 fontsize = 6,
                 x = 0.5,
                 y = 0.99,
                 weight = 'bold')
    
    for i in range(n_DR):
        
        hue = [hue_true, hue_pred[i]]
        titles = [dim_red_names[i], dim_red_names[i] + f' ({clustering_names[i]})']

        for j in range(2):
        
            sns.scatterplot(x = x[i],
                            y = y[i],
                            hue = hue[j],
                            legend = True,
                            palette = list(color_palette[np.unique(hue[j])]),
                            s = 1.4,
                            ax = axes[i][j])
            
            axes[i][j].set_title(titles[j], fontsize = 5)
            
            axes[i][j].tick_params(axis='both',
                                   which='major', 
                                   labelsize = 4, 
                                   width=0.3,
                                   length = 1.7)
            
            axes[i][j].set_xlabel("")
            axes[i][j].set_ylabel("")
            
            for spine in axes[i][j].spines.values():
                spine.set_linewidth(0.3)
            
            axes[i][j].legend(bbox_to_anchor=(1.02, 0.5),
                              loc='center left',
                              borderaxespad=0,
                              fontsize = 4,
                              markerscale = 0.25)
    
    plt.savefig(path, format = 'png', dpi = 300)
    
    
def get_confusion_matrix(dim_red_names, y_true, y_pred, vmin, vmax, w,
                         suptitle, colormap, path):
    
    n_DR = len(dim_red_names)
    
    pred_labels = [np.unique(y_pred[i]).shape[0] for i in range(n_DR)]
    width_ratios = [i/sum(pred_labels) for i in pred_labels]
    
    fig, axes = plt.subplots(2, n_DR, figsize = (w,2.8),
                             gridspec_kw = {'width_ratios': width_ratios,
                                            'height_ratios': [0.9,0.1]})

    plt.subplots_adjust(bottom = 0,
                        top = 0.92,
                        left = 0.05*(6/w),
                        right = 1-0.02*(6/w),
                        wspace = 0.2*(6/w),
                        hspace = 0.18)
           
    fig.suptitle(suptitle,
                 fontsize = 6,
                 x = 0.52,
                 y = 0.98,
                 weight = 'bold')
    
    for i in range(n_DR):
        
        labels = list(set(np.unique(y_true)) | set(np.unique(y_pred[i])))
    
        CM = confusion_matrix(y_true = y_true,
                              y_pred = y_pred[i])
    
        DF_CM = pd.DataFrame(data = CM,
                             columns = labels,
                             index = labels)
        
        DF_CM = DF_CM.loc[list(set(np.unique(y_true))),
                          list(set(np.unique(y_pred[i])))]
        
        # sns.heatmap(DF_CM,
        #             vmin = vmin,
        #             vmax = vmax,
        #             fmt = '3d',
        #             cmap = colormap,
        #             linewidth = .8,
        #             annot = True,
        #             annot_kws={"fontsize": 3},
        #             square = True,
        #             cbar = False,
        #             ax = axes[0][i])

        px.imshow(
            DF_CM
        )
        
        axes[0][i].set_title(dim_red_names[i], fontsize = 5)
        
        axes[0][i].tick_params(axis='both',
                               which='major', 
                               labelsize = 3, 
                               width=0.3,
                               length = 1.7,
                               rotation = 0)
    
        axes[0][i].set_xlabel("")
        axes[0][i].set_ylabel("")
        
        for spine in axes[0][i].spines.values():
            spine.set_linewidth(0.3)
            
    axes[0][0].set_ylabel("Real cluster", fontsize = 5)        
            
    # mappable = axes[0][n_DR-1].get_children()[0]
    # cbar = plt.colorbar(mappable,
    #                     ax = axes[1][:],
    #                     orientation = 'horizontal',
    #                     fraction = 0.1,
    #                     aspect = 150*(w/6),
    #                     location = 'top')
    # cbar.set_ticks([0,250,500,750,1000])
    # cbar.ax.tick_params(labelsize = 4,
    #                     width=0.3,
    #                     length = 1.7,
    #                     rotation = 0)
    # cbar.ax.xaxis.set_ticks_position('bottom')
    # cbar.outline.set_linewidth(0.3)
    # cbar.set_label('Predicted cluster', fontsize = 5, labelpad = 10)
    
    axes[1][0].set_visible(False)
    axes[1][1].set_visible(False)
    axes[1][2].set_visible(False)
    
    plt.savefig(path, format = 'png', dpi = 300)
    