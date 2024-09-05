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
