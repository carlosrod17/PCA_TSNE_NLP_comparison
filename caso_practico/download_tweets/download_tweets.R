setwd("~/Universidad/MasterIngenieriaMatematicaUCM/ASIGNATURAS/TFM/TWITTER")

#install.packages("rtweet")
#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("tidytext")
#install.packages("igraph")
#install.packages("ggraph")

library(rtweet)
library(dplyr)
#library(ggplot2)
#library(tidytext)
#library(igraph)
#library(ggraph)

tweets <- search_tweets(q = "", 
                        n = 50000,
                        lang = "es",
                        result_type='popular',
                        include_rts = FALSE,
                        retryonratelimit = TRUE)
tweets <- tweets[,c(4)]
tweets <- distinct(tweets)

path <- "tweets/tweets_2022_11_22.csv"

write.csv2(tweets,file=path) 