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


topic1 <- c('Ucrania guerra', 'Ucrania Rusia', 'Putin guerra', 'Putin nuclear',
            'Ucrania invasión', 'sanciones Rusia', 'OTAN Rusia', 'Zelenski',
            'Ucrania fosas', 'Rusia crímenes guerra', 'bomba nuclear',
            'Trump guerra Rusia', 'Rusia movilización', 'tropas Rusia',
            'movilización Rusia','avance Ucrania', 'Nord Stream')

topic2 <- c('Qatar', 'mundial Qatar', 'mundial 2022','fútbol Qatar',
            'homosexualidad Qatar', 'construccion estadio mundial',
            'brazalete selección mundial', 'LGTBI Qatar', 'FIFA corrupción', 
            'polémica Qatar muertos', 'gais Qatar', 'gays Qatar',
            'esposas Qatar', 'cárceles Qatar', 'Derechos Humanos mundial',
            '"Derechos humanos" Qatar', 'DDHH Qatar',
            'homosexual "daño mental" Qatar', 'falsos aficionados', 
            '#Qatar2022', 'Qatar cerveza')

topic3 <- c('coronavirus', 'COVID', 'vacuna', 'Pfizer vacuna', 'AstraZeneca', 
            'COVID gripe', 'pandemia', 'variantes covid', 'confinamiento',
            'mascarillas', 'restricciones covid', 'vacunación', 'primera ola',
            'segunda ola', 'tercera ola', 'cuarta ola', 'quinta ola', 
            'sexta ola', 'residencias COVID')

topic4 <- c('crisis energética', 'crisis energía', 'corbata Sánchez',
            'inflación', 'recesión', 'precio gas', 'IPC',
            'precio energía', '"independencia energética"',
            'dependencia gas', 'energía nuclear crisis', 'UE gas',
            'Alemania gas', 'encarecimiento energía', 'encarecimiento gas',
            'invierno Europa', 'precio calefacción')

topic5 <- c('inmigración España', 'menas', 'Melilla BBC',
            'valla Melilla', 'inmigrantes ilegales España',
            'inmigración Europa', 'pateras', 'Marlaska Melilla', 
            'inmigrantes Melilla', 'tragedia Melilla', 'tragedia Melilla',
            'Melilla BBC', 'Ceuta Marruecos', 'Melilla Marruecos',
            'delitos extranjeros', 'España Marruecos valla',
            'ministerio interior Melilla', 'migrantes muertos', 
            'mafias inmigración', 'Ceuta Melilla', 'inmigración Barcelona',
            'inmigrantes España', 'inmigrantes Barcelona',
            'inmigrantes Valencia', 'menas Batán', 'inmigrantes Europa',
            '"inmigración marroquí"')

topic6 <- c('mujeres iraníes', 'mujeres Irán', 'machismo Irán',
            'revolución hijab', 'protestas Irán', '"Mahsa Amini"',
            'política Irán', 'sanciones Irán', 'represión Irán', 'DDHH Irán',
            '#IranProtests2022', '#IranRevolution2022', 'clérigos Irán',
            'condena muerte Irán', '"Derechos Humanos" Irán', 
            'manifestante Irán', 'dictadura Irán', 'velo Irán',
            '"pena de muerte" Irán', 'Islám Irán',
            'feministas Irán', 'activistas iraníes', 'mujeres velo',
            'gobierno Irán', '#IranRevolution', 'solidaridad Irán',
            'turbante Irán', 'líderes religiosos Irán')

topic7 <- c('LGTBI SEPE', 'Ley trans', 'ley "sí es sí"', 'justicia machista',
            'ministerio igualdad', 'niños trans', 'médicos trans',
            'trans deporte femenino', 'hormonación trans', 'registro trans',
            'Irene Montero', '"lenguaje inclusivo" Irene Montero',
            '"fascistas con toga', '"cambio de sexo"', 'reducción condena ley',
            '"ideología de género"', '"hazte oir"')

topic8 <- c('correos comunista', 'sello PCE', 'sello comunista',
           'abogados cristianos sello', '"abogados cristianos" correos', 
           'sello "partido comunista"', 'juez sello PCE', 'centenario PCE',
           'centenario comunista', 'correos neutralidad PCE')

topic9 <- c('sanidad Madrid', 'sanidad publica Ayuso',
            'sanidad profesionales huelga', '#SanidadPublica', 
            '#MadridSeLevantaEl13', 'sanitarios Madrid',
            'sanitarios Ayuso', 'huelga sanidad', 'ambulatorios Madrid',
            'atención primaria Madrid', 'marea blanca Madrid',
            'Ayuso sanidad', 'recortes sanidad PP', 'sanidad madrileña')

topic10 <- c('delito sedición', 'independentismo', 'líderes independentistas',
             'reforma sedición', 'castellano aulas Cataluña',
             'Cataluña castellano', 'Cataluña referendum', 'sedición 1-O',
             'Cataluña 2017', 'Puigdemont', 'Junqueras')


topics = list(topic1,topic2,topic3,topic4,topic5,
              topic6,topic7,topic8,topic9,topic10)

ntweets = c(800,750,500,750,3500,
            2500,1000,2000,750,1000)

for (i in 1:length(topics)) {
  tweets <- search_tweets2(q = topics[[i]], 
                           n = ntweets[i],
                           lang = "es",
                           result_type='popular',
                           include_rts = FALSE,
                           retryonratelimit = TRUE)
  tweets <- tweets[,c(4)]
  tweets <- distinct(tweets)
  
  path <- paste("tweets_by_class_2022_11_19/tweets_",as.character(i),".csv",
                sep="")
  
  write.csv2(tweets,file=path) 
  
}





