import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy

from nltk.stem.snowball import SnowballStemmer

from langdetect import detect
from langdetect import DetectorFactory

from difflib import SequenceMatcher

from gensim.models import phrases

from collections import Counter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix

from mip import Model, xsum, maximize

from var_def import seed


def NLP_preprocessing(text, stop_chars, stop_words):
    
    """
    
    NLP_preprocessing performs the first step of text processing. It tokenizes
    the text, normalizes its tokens to lowercase, excludes URLs, hashtags and
    user mentions, and removes stopwords and short tokens.

    Input:
        - text (str): The text to be preprocessed.
        
        - stop_chars (str): Special characters to be removed from the text.
        
        - stop_words (list of str): List of stopwords to be excluded from the
          result.

    Output:
        - new_text (str): The preprocessed text.
        
        
    """
    
    text = " ".join(text.split())
    text = text.lower()
    
    text = re.sub(r"(?:\@|https?\://|www.|#)\S+", " ", text)
    
    table = str.maketrans(stop_chars, " "*len(stop_chars))
    text = text.translate(table)
    
    text = re.sub(r'([a-záéíóú])\1{2,}', r'\1', text)
    
    tokens = text.split()
    tokens = [token for token in tokens if len(token)>2]
    tokens = [token for token in tokens if token not in stop_words]
    
    new_text = " ".join(tokens)
    
    return new_text


def NLP_language_filtering(dataframe, var_name, language, seed):
    
    """
    
    NLP_language_filtering takes a corpus of texts in a dataframe and
    processes each text using a language detection model. Any text whose
    predicted language doesn't match the required language is removed from
    the dataframe. 

    To learn more about the language detection model used, see:
        https://pypi.org/project/langdetect/

    Input:
        - dataframe (Pandas DataFrame): The dataframe containing the corpus
          of texts.
          
        - var_name (str): The name of the variable in the dataframe that
          contains the corpus of texts.
          
        - language (str): The ISO 639-1 code representing the required
          language for the texts to be kept in the corpus.
          
        - seed (int): A seed to control the generation of random numbers.

    Output:
        - new_dataframe (Pandas DataFrame): The dataframe containing the
          texts whose predicted language matches the required language.
          
    """
    
    index_to_delette = []
    
    for ind in dataframe.index:
        
        try:
            
            DetectorFactory.seed = seed
            text_language = detect(dataframe.loc[ind,var_name])
            
        except:
            
            text_language = None
        
        if text_language is None or language != text_language:
        
            index_to_delette.append(ind)
            
    new_dataframe = dataframe.drop(index_to_delette, axis = 0)
    
    return new_dataframe            


def NLP_lemmatizing(text,
                    stop_words,
                    nlp = spacy.load('es_core_news_md')):

    """
    
    NLP_lemmatizing performs morphological filtering and lemmatization on the
    tokens of a text. It also removes stopwords from the result.

    To learn more about the model used to tag and lemmatize the tokens, see:
        https://spacy.io/models/es

    Inputs:
        - text (str): The text to be processed.
        
        - stop_words (list of str): List of stopwords to be excluded from the
          result.

    Output:
        - new_text (str): The processed text.
        
        
    """  
    
    doc = nlp(text)
    
    doc = [token for token in doc if token.pos_ in ['ADJ', 'NOUN', 'PROPN', 'VERB']]
    
    lemmas = [str(token.lemma_).split()[0] if " " in str(token.lemma_) else str(token.lemma_) for token in doc]
    lemmas = [lemma for lemma in lemmas if lemma not in stop_words]
    
    new_text = " ".join(lemmas)
    new_text = re.sub(r"á|à|ä|â|ã","a", new_text)
    new_text = re.sub(r"é|è|ë|ê","e", new_text)
    new_text = re.sub(r"í|ì|ï|î","i", new_text)
    new_text = re.sub(r"ó|ò|ö|ô|õ","o", new_text)
    new_text = re.sub(r"ú|ù|ü|û","u", new_text)
    
    return new_text


def NLP_stemming(text,
                 stemmer = SnowballStemmer('spanish')):
    
    """
    
    NLP_stemming applies semantic pruning to the tokens of a text.
    
    To learn more about the model used for token pruning, see:
        https://www.nltk.org/_modules/nltk/stem/snowball.html
        
    Input:
        - text (str): The text to be processed.
        
    Output:
        - new_text (str): The processed text.
        
    
    """
    
    tokens = text.split()
    roots = [stemmer.stem(token) for token in tokens]
    
    new_text = " ".join(roots)
    
    return new_text


def NLP_translate(text, dictionary):
    
    """
    
    NLP_translate performs translation based on a given dictionary. It
    replaces occurrences of any key in the dictionary with the corresponding
    value in the text.

    Inputs:
        - text (str): The text to be translated.
        
        - dictionary (dict): A dictionary where the keys and values are
          strings. The keys are searched in the text and translated to their
          respective values.

    Output:
        - new_text (str): The translated text.
        
        
    """
    
    tokens = text.split()
    tokens = [dictionary[token] if token in dictionary.keys() else token for token in tokens]
    
    new_text = " ".join(tokens)
    
    return new_text


def NLP_similarity_translation(dataframe, var_name, min_similarity):
    
    """
    
    NLP_similarity_translation extracts the set of tokens from a corpus of 
    texts and processes many pairs of tokens using a string similarity model.
    Those pairs of tokens that obtain a score greater or equal to a threshold
    are included in a dictionary as key-value pairs. Once this process is
    finished, the corpus receives a translation of tokens based on the built
    dictionary.
    
    To learn more about the string similarity model used, see:
        https://docs.python.org/es/3/library/difflib.html
    
    Inputs:
        - dataframe (Pandas DataFrame): The dataframe containing the corpus
          of texts.
        - var_name (str): The name of the variable in the dataframe that
          contains the corpus of texts.
        - min_similarity (float): The threshold used to decide which pairs of
          tokens are included in the dictionary. It should be between 0 and 1.
    
    Output:
        - new_dataframe (Pandas DataFrame): The dataframe translated.
        
        
    """
    
    voc = []
    
    for text in dataframe[var_name]:
        voc.extend(text.split())
        
    voc = sorted(list(set(voc)))
    
    dictionary = {}
    
    for i in range(len(voc)-10):
        
        for j in range(i+1,i+10):
            
            if SequenceMatcher(None, voc[i], voc[j]).ratio()>= min_similarity:
                
                dictionary[voc[i]] = voc[j]
                break
            
    for value in dictionary.values():
        
        if value in dictionary.keys():
            
            keys = [key for key in dictionary.keys() if dictionary[key] == value]
            
            for key in keys:
                
                dictionary[key] = dictionary[value]
                
    new_dataframe = dataframe[var_name].apply(lambda x: NLP_translate(x,dictionary))
            
    return new_dataframe


def NLP_bigrams_detection(dataframe, var_name, min_count, threshold, scoring):
    
    """
    NLP_bigrams_detection processes a set of sentences extracted from a corpus
    of texts using a bigram detection model. This model considers every pair of
    tokens that appear together in the set of sentences and selects those that
    appear more frequently than expected based on parameters provided by the
    user. When these pairs of tokens are selected, a bigram is constructed with
    a delimiter in between, and the sentences are translated by replacing the
    pairs of tokens with the corresponding bigrams.
    
    To learn more about the bigram detection model used, see: 
        https://radimrehurek.com/gensim/models/phrases.html
    
    Input:
        - dataframe (Pandas DataFrame): The dataframe containing the corpus of
          texts.
          
        - var_name (str): The name of the variable in the dataframe that
          contains the corpus of texts.
          
        - parameters (tuple): A tuple of parameters to control the bigram
          detection model. parameters[0] represents the minimum frequency
          required for a pair of tokens to be considered as a candidate for
          a bigram. It should be an integer greater than 0. parameters[1]
          represents the score that a candidate must achieve to be considered
          a bigram. It should be a float between 0 and 1.
          
        - delimiter (str): The string used to concatenate a pair of tokens
          into a bigram.
        
    Output:
        - new_dataframe (Pandas DataFrame): The translated dataframe.
        
        
    """

    corpus = dataframe[var_name]

    sentences = [text.split() for text in corpus]

    logging.getLogger().setLevel(logging.ERROR)
    
    bigram_model = phrases.Phrases(sentences,
                                   min_count = min_count,
                                   threshold = threshold,
                                   scoring = scoring,
                                   delimiter = "_")
    
    logging.getLogger().setLevel(logging.INFO)
    
    """    

    bigrams = bigram_model.export_phrases().items()
    bigrams = sorted(bigrams, key = lambda x: x[1], reverse = True)
    
    for bigram in bigrams:
        
        f1 = bigram_model.vocab[bigram[0].split("_")[0]]
        f2 = bigram_model.vocab[bigram[0].split("_")[1]]
        
        print(bigram[0],bigram[1],bigram_model.vocab[bigram[0]],f1,f2)
    
    print("__________________________________")
    
    for bigram in bigrams:
        
        f1 = bigram_model.vocab[bigram[0].split("_")[0]]
        f2 = bigram_model.vocab[bigram[0].split("_")[1]]
        
        if max(f1,f2)/min(f1,f1)<3:
            print(bigram[0],bigram[1],bigram_model.vocab[bigram[0]],f1,f2)
    
    """
    
    new_dataframe = [" ".join(i) for i in bigram_model[sentences]]
        
    return new_dataframe


def NLP_frequency_threshold(dataframe, var_name, threshold):
    
    """
    
    NLP_frequency_threshold extracts the tokens from a corpus of text and
    their absolute frequencies. Tokens whose frequency does not meet a
    specified threshold will be removed from all the texts in the corpus.
    
    Input:
        - dataframe (Pandas DataFrame): The dataframe containing the corpus of
          texts.
          
        - var_name (str): The name of the variable in the dataframe that
          contains the corpus of texts.
          
        - threshold (int): The frequency threshold that a token must meet in
          order to be kept in the corpus.
        
    Output:
        - new_dataframe (Pandas DataFrame): The dataframe without the least
          frequent tokens.
          
          
    """
    
    corpus = dataframe[var_name]
    
    voc = []
    
    for text in corpus:
        voc.extend(text.split())
        
    freq = Counter(voc)
    
    words_to_delette = [word for word, frequency in freq.items() if frequency < threshold]
    
    new_corpus = []
    
    for i in range(len(corpus)):
        tokens = corpus[i].split()
        tokens = [token for token in tokens if token not in words_to_delette]
        
        new_corpus.append(" ".join(tokens))
    
    return new_corpus


def NLP(dataframe, var_name, params):
    
    """
    
    NLP takes a corpus of texts in a dataframe and transforms this corpus
    through different NLP steps, including preprocessing, language filtering,
    tag filtering, lemmatizing, stemming, token embedding, removal of least
    frequent tokens, and bigram detection. The results of each of these steps
    are added to the original dataframe as new variables.
    
    Input:
        - dataframe (Pandas DataFrame): it contains the corpus of texts.
        
        - var_name (str): The name of the variable in the dataframe that
          contains the corpus of texts.
          
        - min_similarity (float): A parameter of the NLP_similarity_translation
          function. It should be between 0 and 1.
          
        - bigrams_parameters (tuple): Parameters of the NLP_bigrams_detection
          function. bigrams_parameters[0] should be an integer greater than 0,
          and bigrams_parameters[1] should be a float between 0 and 1.
          
        - df_min (int): A parameter of the NLP_frequency_threshold function.
          It should be an integer greater than 0.
    
    Output:
        - dataframe (Pandas DataFrame): Contains the results of each step of
          the NLP process. These results appear in variables called 'LIST_1',
          'LIST_2', 'LIST_3', 'LIST_4', 'LIST_5', 'LIST_6'. If the original
          dataframe had a variable with one of these names, it will be replaced
          in the output.
          
        
    """
    
    n_texts = dataframe.shape[0]
    
    # STEP 1: PREPROCESING (CLEANING TEXTS, STOPWORDS REMOVAL)
    
    logging.info("    Preprocessing texts.")
    logging.info("       - Stopchars to be deletted:")
    [logging.info(f"          {list(params['stopchars'])[15*i:(15*(i+1))]}") for i in range(len(params["stopchars"])//15) ]
    logging.info(f"          {list(params['stopchars'])[-(len(params['stopchars'])%15):]}")
    logging.info("       - Stopwords to be deletted:")
    [logging.info(f"          {params['stopwords'][6*i:(6*(i+1))]}") for i in range(len(params["stopwords"])//6) ]
    logging.info(f"          {list(params['stopwords'])[-(len(params['stopwords'])%6):]}")
    
    dataframe['LIST_1'] = dataframe[var_name].apply(
        lambda x: NLP_preprocessing(
            x,
            stop_chars = params["stopchars"],
            stop_words = params["stopwords"]
        )
    )
    
    dataframe.drop_duplicates(['LIST_1'], keep = 'first', inplace = True)
    
    logging.info("    Preprocessing of texts performed.")
    logging.info(f"       + There are {n_texts - dataframe.shape[0]} texts that have been discarded.")
    logging.info(f"       + There are {dataframe.shape[0]} texts left.")
    
    n_texts = dataframe.shape[0]
    
    # STEP 2: LANGUAGE FILTERING
    
    logging.info("    Filtering texts by language with parameters:")
    logging.info(f"       - language = {params['langauge']}")
    logging.info(f"       - seed = {params['seed']}")
    
    dataframe = NLP_language_filtering(dataframe = dataframe,
                                       var_name = 'LIST_1',
                                       language = params["langauge"],
                                       seed = params["seed"])
    
    logging.info("    Filtering by language performed.")
    logging.info(f"       + There are {n_texts - dataframe.shape[0]} texts that have been discarded.")
    logging.info(f"       + There are {dataframe.shape[0]} texts left.")
    
    n_texts = dataframe.shape[0]
    
    # STEP 3: TAG FILTERING AND LEMMATIZING
    
    logging.info("    Tag filtering and lemmatizing.")
    
    dataframe['LIST_2'] = dataframe['LIST_1'].apply(lambda x: NLP_lemmatizing(x, stop_words = params["stopwords"]))
    
    dataframe.drop_duplicates(['LIST_2'], keep = 'first', inplace = True)
    
    logging.info("    Tag filtering and lemmatizing performed.")
    logging.info(f"       + There are {n_texts - dataframe.shape[0]} texts that have been discarded.")
    logging.info(f"       + There are {dataframe.shape[0]} texts left.")
    
    n_texts = dataframe.shape[0]
    
    # STEP 4: STEMMING
    
    logging.info("    Stemming.")
    
    dataframe['LIST_3'] = dataframe['LIST_2'].apply(NLP_stemming)
    
    dataframe.drop_duplicates(['LIST_3'], keep = 'first', inplace = True)
    
    logging.info("    Steming performed.")
    logging.info(f"       + There are {n_texts - dataframe.shape[0]} texts that have been discarded.")
    logging.info(f"       + There are {dataframe.shape[0]} texts left.")
    
    n_texts = dataframe.shape[0]
    # STEP 5: TOKEN EMBEDDING BY SIMILARITY
    
    logging.info("    Token embedding by similarity.")
    
    dataframe['LIST_4'] = NLP_similarity_translation(
        dataframe = dataframe,
        var_name = 'LIST_3', 
        min_similarity = params["min_similarity"]
    )
    
    dataframe.drop_duplicates(['LIST_4'], keep = 'first', inplace = True)
    
    logging.info("    Token embedding by similarity performed.")
    logging.info(f"       + There are {n_texts - dataframe.shape[0]} texts that have been discarded.")
    logging.info(f"       + There are {dataframe.shape[0]} texts left.")
    
    n_texts = dataframe.shape[0]
    
    # STEP 6: LEAST FREQUENCY TOKENS REMOVAL
    
    logging.info("    Least frequent tokens removal with parameters:")
    logging.info(f"       - min_frequency = {params['min_frequency']}")
    
    dataframe['LIST_5'] = NLP_frequency_threshold(
        dataframe = dataframe,
        var_name = 'LIST_4',
        threshold = params["min_frequency"]
    )
    
    dataframe.drop_duplicates(['LIST_5'], keep = 'first', inplace = True)
    
    logging.info("    Least frequent tokens removal performed.")
    logging.info(f"       + There are {n_texts - dataframe.shape[0]} texts that have been discarded.")
    logging.info(f"       + There are {dataframe.shape[0]} texts left.")
    
    n_texts = dataframe.shape[0]
    
    # STEP 7: BIGRAMS DETECTION
    
    logging.info("    Bigrams detection.")
    logging.info(f"       - min_count = {params['bigrams_min_count']}")
    logging.info(f"       - min_count = {params['bigrams_threshold']}")
    logging.info(f"       - min_count = {params['bigrams_scoring']}")
    
    dataframe['LIST_6'] = NLP_bigrams_detection(
        dataframe = dataframe,
        var_name = 'LIST_5',
        min_count = params["bigrams_min_count"],
        threshold = params["bigrams_threshold"],
        scoring = params["bigrams_scoring"]
    )
    
    logging.info("    Bigrams detection performed.")
    logging.info(f"       + There are {n_texts - dataframe.shape[0]} texts that have been discarded.")
    logging.info(f"       + There are {dataframe.shape[0]} texts left.")
    
    n_texts = dataframe.shape[0]
    
    return dataframe    


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
                            init = 'k-means++',
                            n_init = 5,
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
        
        print('Something went wrong. The input model iss not available.')


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
        
        sns.heatmap(DF_CM,
                    vmin = vmin,
                    vmax = vmax,
                    fmt = '3d',
                    cmap = colormap,
                    linewidth = .8,
                    annot = True,
                    annot_kws={"fontsize": 3},
                    square = True,
                    cbar = False,
                    ax = axes[0][i])
        
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
            
    mappable = axes[0][n_DR-1].get_children()[0]
    cbar = plt.colorbar(mappable,
                        ax = axes[1][:],
                        orientation = 'horizontal',
                        fraction = 0.1,
                        aspect = 150*(w/6),
                        location = 'top')
    cbar.set_ticks([0,250,500,750,1000])
    cbar.ax.tick_params(labelsize = 4,
                        width=0.3,
                        length = 1.7,
                        rotation = 0)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.outline.set_linewidth(0.3)
    cbar.set_label('Predicted cluster', fontsize = 5, labelpad = 10)
    
    axes[1][0].set_visible(False)
    axes[1][1].set_visible(False)
    axes[1][2].set_visible(False)
    
    plt.savefig(path, format = 'png', dpi = 300)
    