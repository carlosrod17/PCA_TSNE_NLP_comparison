##############################################################################
# CREATION AND LOAD OF THE DATASET
##############################################################################

import os
import sys
import logging
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

from var_def import TFM_PATH
from var_def import BBDD_PATH_1
from var_def import BBDD_PATH_2
from var_def import delimiter
from var_def import n_clusters
from var_def import NLP_parameters
from var_def import vmax
from var_def import vmin
from var_def import seed

from utils.natural_language_processing import NLP

logging.basicConfig(
    filename = os.path.join(TFM_PATH, sys.argv[1]),
    format="%(asctime)s %(levelname)s: %(message)s",
    level = logging.INFO)

logging.info("-"*120)
logging.info("Starting processing.")
logging.info("Loading tweets.")

DF_tweets = (
    pd.concat(
        [
            pd.read_csv(
                os.path.join(BBDD_PATH_1, f"tweets_{i}.csv"),
                sep = ";",
                header = None,
                names = ["TWEET_NUM","TWEET"],
                dtype = str
            )
            .drop_duplicates(["TWEET"], keep = "first")
            .assign(TWEET_ID = lambda _df: f"{i:02d}_" + _df["TWEET_NUM"])
            .assign(CLUSTER_REAL = f"{i:02d}")
            .set_index("TWEET_ID")
            .filter(["CLUSTER_REAL", "TWEET"])
            .assign(
                TWEET = lambda df_: df_.TWEET.apply(lambda x: x.replace(delimiter, ""))
            )
            
            for i in range(1,n_clusters+1)
            
        ]
    )
    .drop_duplicates(["TWEET"], keep = False)
)

logging.info(f"{DF_tweets.shape[0]:06d} tweets have being loaded.")


# NLP

logging.info("Processing tweets.")

DF_tweets_processed = NLP(
    dataframe = DF_tweets,
    var_name = 'TWEET',
    params = NLP_parameters
)

logging.info(f"Tweets processed. There are {DF_tweets_processed.shape[0]:06d} left after processing.")


# RESAMPLING

logging.info("Resamling dataset.")

for value in range(vmax, vmin, -10):
    
    logging.info(f"   Searching for {value:04d} tweets of each group.")
    
    try:
        
        DF_tweets_resampled = (
            pd.concat(
                [
                    resample(
                        DF_tweets_processed[DF_tweets_processed['CLUSTER_REAL']==f"{i:02d}"],
                        replace = False,
                        n_samples = vmax,
                        random_state = seed
                    )
                
                    for i in range(1,n_clusters+1)   
                
                ] 
            )        
        )
        
        logging.info("      + Resampling has finished succesfully.")
        break
    
    except:

        logging.info(f"      + Resampling failed. There are not {value:04d} tweets of each group.")
 

# TFIDF COMPUTATION

logging.info("Computing tfidf vectorization.")

tfidf = TfidfVectorizer()

DF_tfidf = (
    pd.DataFrame(
        tfidf
        .fit_transform(DF_tweets_resampled['LIST_5'])
        .todense(),
        columns = sorted(tfidf.vocabulary_, key = tfidf.vocabulary_.get)
    )
    .assign(TWEET_ID = DF_tweets_resampled.index) 
    .set_index("TWEET_ID")
)

logging.info(f"tfidf vectorization finished with {len(tfidf.vocabulary_):05d} tokens.")


# MOST FREQUENT WORDS PER CLUSTER_REAL

logging.info("Calculating most frequent words per cluster.")

DF_top_words = pd.DataFrame(
    DF_tfidf
    .assign(CLUSTER = lambda df_: df_.index.astype(str).str[0:2])
    .groupby("CLUSTER").sum()
    .apply(lambda row: pd.Series(row.nlargest(5).index.to_list()), axis = 1)
    .reset_index()
    .rename(columns = {i: f"top_word_{i+1}" for i in range(5)})
)

logging.info(f"Top 5 words per cluster are:")
[logging.info(f"   + CLUSTER {row[0]}: {[row[j] for j in range(1,len(row))]}") for i, row in DF_top_words.iterrows()]

# SAVE RESULTS

logging.info("Saving processed data.")

DF_tweets.to_csv(os.path.join(BBDD_PATH_2, "tweets_preprocessed.csv.gz"), sep = delimiter, compression = "gzip")
DF_tweets_processed.to_csv(os.path.join(BBDD_PATH_2, "tweets_processed.csv.gz"), sep = delimiter, compression = "gzip")
DF_tweets_resampled.to_csv(os.path.join(BBDD_PATH_2, "tweets_resampled.csv.gz"), sep = delimiter, compression = "gzip")
DF_tfidf.to_csv(os.path.join(BBDD_PATH_2, "tweets_tfidf.csv.gz"), sep = delimiter, compression = "gzip")
DF_top_words.to_csv(os.path.join(BBDD_PATH_2, "top_words.csv.gz"), sep = delimiter, index = False, compression = "gzip")

logging.info("Processing finished.")
logging.info("-"*120)
