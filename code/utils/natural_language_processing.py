import logging
import re
import spacy

from nltk.stem.snowball import SnowballStemmer

from langdetect import detect
from langdetect import DetectorFactory

from difflib import SequenceMatcher

from gensim.models import phrases

from collections import Counter


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