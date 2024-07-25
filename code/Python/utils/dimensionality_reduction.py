import logging
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from var_def import seed


def fit_transform_model(data, model, params):
    
    """
    
    fit_transform_model implements a dimensionality reduction model (selected
    from a set of available models) on a dataset, embedding it in a 2D space.
    The model's training parameters are set and the seed is fixed to ensure
    replicability.
    
    Input:
        - model (str): should be one of the following: 'PCA', 'TSNE', 
          'PCA + LLE', 'PCA + TSNE'. If not, a warning will be printed.
        
        - data (NumPy Array or Pandas DataFrame): data set that will be
          reduced using the selected model.
          
    Output:
        - MAT_embedded (NumPy Array): array of dimensions (n_samples,2), where
          n_samples is the number of points in the original dataset (number of
          rows in data).
    
    
    """
    
    if model == 'PCA':
        
        logging.info(f"    Building {model} model with parameters:")
        [logging.info(f"       - {key} = {value}") for key,value in params.items()]
        
        pca = PCA(n_components = params["n_components"],
                  svd_solver = params["svd_solver"],
                  random_state = params["seed"])
        
        logging.info("    Embedding data with PCA model.")
        
        MAT_embedded = pca.fit_transform(data)
        
        logging.info("    Data embedded succesfully.")
        
        return MAT_embedded
        
    elif model == 'TSNE':
        
        MAT_aux = data
        
        n_executions = len(params)
        
        for i, execution in enumerate(sorted(params.keys())):
            
            logging.info(f"    Building {model} (execution {i}) model with parameters:")
            [logging.info(f"       - {key} = {value}") for key,value in params[execution].items()]
            
            
            tsne = TSNE(n_components = params[execution]["n_components"],
                        init = params[execution]["init"],
                        perplexity = params[execution]["perplexity"],
                        early_exaggeration = params[execution]["early_exageration"],
                        learning_rate = params[execution]["learning_rate"],
                        n_iter = params[execution]["n_iter"],
                        n_iter_without_progress = params[execution]["n_iter_without_progress"],
                        n_jobs = params[execution]["n_jobs"],
                        random_state = params[execution]["random_state"],
                        verbose = params[execution]["verbose"])
            
            logging.info("    Embedding data with TSNE model.")
            
            MAT_aux = tsne.fit_transform(MAT_aux)
            
            logging.info("    Data embedded succesfully.")
            logging.info(f"       + Number of iterations: {tsne.n_iter_}")
            logging.info(f"       + Kullback-Leibler divergence: {tsne.kl_divergence_}")
        
        MAT_embedded = MAT_aux
        
        return MAT_embedded
        
    elif model == 'PCA_TSNE':
        
        logging.info("    Searching for optim value of components.")
        logging.info("    Initializing PCA model with parameters:")
        [logging.info(f"       - {key} = {value}") for key,value in params["PCA"].items()]
        
        
        pca_aux = PCA(n_components = params["PCA"]["n_components"],
                      svd_solver = params["PCA"]["svd_solver"],
                      random_state = params["PCA"]["seed"])
        
        logging.info("    Embedding data with PCA model.")
        
        MAT_PCA_aux = pca_aux.fit_transform(data)
        
        logging.info("    Data embedded succesfully.")        
        logging.info("    Computing cumulative variance ratios.")
        
        variance_ratios = pca_aux.explained_variance_ratio_
        cumulative_variance_ratios = np.cumsum(variance_ratios)
        
        logging.info(f"    Selecting as candidates those which have cumulative variance ratio between {params['cumulative_variance_ratio']['min']} and {params['cumulative_variance_ratio']['max']}.")
        
        n_PCA_min = np.where(cumulative_variance_ratios > params["cumulative_variance_ratio"]["min"])[0][0]
        n_PCA_max = np.where(cumulative_variance_ratios > params["cumulative_variance_ratio"]["max"])[0][0]-1
        
        logging.info(f"    Choosing optim value among {list(range(n_PCA_min, n_PCA_max+1))}.")
        
        n_clusters_list = range(max(2,n_PCA_min), n_PCA_max+1)
        best_s_score = -1
        optim_PCA = 500
        
        for k in n_clusters_list:
        
            kmeans = KMeans(n_clusters = k,
                            init = 'random',
                            n_init = 50,
                            random_state = seed)
            
            kmeans.fit(MAT_PCA_aux)
            
            s_score = silhouette_score(MAT_PCA_aux,
                                       kmeans.labels_)
            
            logging.info(f"       + K-Means with {k:02d} components gets {s_score:7.5f} of silhouette score.")
            
            if s_score > best_s_score:
                optim_PCA = np.unique(kmeans.labels_).shape[0]
                best_s_score = s_score

        logging.info(f"    Optim value of PCA is {optim_PCA}.")
        
        params["PCA"]["n_components"] = optim_PCA
        
        logging.info("    Building PCA model with parameters:")
        [logging.info(f"       - {key} = {value}") for key,value in params["PCA"].items()]
        
        pca = PCA(n_components = params["PCA"]["n_components"],
                  svd_solver = params["PCA"]["svd_solver"],
                  random_state = params["PCA"]["seed"])
        
        logging.info("    Embedding data with PCA model.")
        
        MAT_aux = pca.fit_transform(data)
        
        n_executions = len(params["TSNE"])
        
        for i, execution in enumerate(sorted(params["TSNE"].keys())):
            
            logging.info(f"    Building TSNE (execution {i}) model with parameters:")
            [logging.info(f"       - {key} = {value}") for key,value in params["TSNE"][execution].items()]
            
            
            tsne = TSNE(n_components = params["TSNE"][execution]["n_components"],
                        init = params["TSNE"][execution]["init"],
                        perplexity = params["TSNE"][execution]["perplexity"],
                        early_exaggeration = params["TSNE"][execution]["early_exageration"],
                        learning_rate = params["TSNE"][execution]["learning_rate"],
                        n_iter = params["TSNE"][execution]["n_iter"],
                        n_iter_without_progress = params["TSNE"][execution]["n_iter_without_progress"],
                        n_jobs = params["TSNE"][execution]["n_jobs"],
                        random_state = params["TSNE"][execution]["random_state"],
                        verbose = params["TSNE"][execution]["verbose"])
            
            logging.info("    Embedding data with TSNE model.")
            
            MAT_aux = tsne.fit_transform(MAT_aux)
            
            logging.info("    Data embedded succesfully.")
            logging.info(f"       + Number of iterations: {tsne.n_iter_}")
            logging.info(f"       + Kullback-Leibler divergence: {tsne.kl_divergence_}")
        
        MAT_embedded = MAT_aux
        
        return MAT_embedded
        
    else:
        
        print('Something went wrong. The input model is not available.')