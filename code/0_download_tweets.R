setwd("/opt/shared")

library(rtweet)
library(dplyr)

topic1 <- c('Ucrania guerra', 'Ucrania Rusia', 'Putin guerra', 'Putin nuclear',
            'Ucrania invasi�n', 'sanciones Rusia', 'OTAN Rusia', 'Zelenski',
            'Ucrania fosas', 'Rusia cr�menes guerra', 'bomba nuclear',
            'Trump guerra Rusia', 'Rusia movilizaci�n', 'tropas Rusia',
            'movilizaci�n Rusia','avance Ucrania', 'Nord Stream')

topic2 <- c('Qatar', 'mundial Qatar', 'mundial 2022','f�tbol Qatar',
            'homosexualidad Qatar', 'construccion estadio mundial',
            'brazalete selecci�n mundial', 'LGTBI Qatar', 'FIFA corrupci�n', 
            'pol�mica Qatar muertos', 'gais Qatar', 'gays Qatar',
            'esposas Qatar', 'c�rceles Qatar', 'Derechos Humanos mundial',
            '"Derechos humanos" Qatar', 'DDHH Qatar',
            'homosexual "da�o mental" Qatar', 'falsos aficionados', 
            '#Qatar2022', 'Qatar cerveza')

topic3 <- c('coronavirus', 'COVID', 'vacuna', 'Pfizer vacuna', 'AstraZeneca', 
            'COVID gripe', 'pandemia', 'variantes covid', 'confinamiento',
            'mascarillas', 'restricciones covid', 'vacunaci�n', 'primera ola',
            'segunda ola', 'tercera ola', 'cuarta ola', 'quinta ola', 
            'sexta ola', 'residencias COVID')

topic4 <- c('crisis energ�tica', 'crisis energ�a', 'corbata S�nchez',
            'inflaci�n', 'recesi�n', 'precio gas', 'IPC',
            'precio energ�a', '"independencia energ�tica"',
            'dependencia gas', 'energ�a nuclear crisis', 'UE gas',
            'Alemania gas', 'encarecimiento energ�a', 'encarecimiento gas',
            'invierno Europa', 'precio calefacci�n')

topic5 <- c('inmigraci�n Espa�a', 'menas', 'Melilla BBC',
            'valla Melilla', 'inmigrantes ilegales Espa�a',
            'inmigraci�n Europa', 'pateras', 'Marlaska Melilla', 
            'inmigrantes Melilla', 'tragedia Melilla',
            'Melilla BBC', 'Ceuta Marruecos', 'Melilla Marruecos',
            'delitos extranjeros', 'Espa�a Marruecos valla',
            'ministerio interior Melilla', 'migrantes muertos', 
            'mafias inmigraci�n', 'Ceuta Melilla', 'inmigraci�n Barcelona',
            'inmigrantes Espa�a', 'inmigrantes Barcelona',
            'inmigrantes Valencia', 'menas Bat�n', 'inmigrantes Europa',
            '"inmigraci�n marroqu�"', 'im�genes tragedia Melilla',
            'videos Melilla')

topic6 <- c('mujeres iran�es', 'mujeres Ir�n', 'machismo Ir�n',
            'revoluci�n hijab', 'protestas Ir�n', '"Mahsa Amini"',
            'pol�tica Ir�n', 'sanciones Ir�n', 'represi�n Ir�n', 'DDHH Ir�n',
            '#IranProtests2022', '#IranRevolution2022', 'cl�rigos Ir�n',
            'condena muerte Ir�n', '"Derechos Humanos" Ir�n', 
            'manifestante Ir�n', 'dictadura Ir�n', 'velo Ir�n',
            '"pena de muerte" Ir�n', 'Isl�m Ir�n',
            'feministas Ir�n', 'activistas iran�es', 'mujeres velo',
            'gobierno Ir�n', '#IranRevolution', 'solidaridad Ir�n',
            'turbante Ir�n', 'l�deres religiosos Ir�n')

topic7 <- c('LGTBI SEPE', 'Ley trans', 'ley "s� es s�"', 'justicia machista',
            'ministerio igualdad', 'ni�os trans', 'm�dicos trans',
            'trans deporte femenino', 'hormonaci�n trans', 'registro trans',
            'Irene Montero', '"lenguaje inclusivo" Irene Montero',
            'fascistas con toga', '"cambio de sexo"', 'reducci�n condena ley',
            '"ideolog�a de g�nero"', '"hazte oir"', 'rebajas pena "s� es s�"')

topic8 <- c('correos comunista', 'sello PCE', 'sello comunista',
            'abogados cristianos sello', '"abogados cristianos" correos', 
            'sello "partido comunista"', 'juez sello PCE', 'centenario PCE',
            'centenario comunista', 'correos neutralidad PCE')

topic9 <- c('sanidad Madrid', 'sanidad publica Ayuso',
            'sanidad profesionales huelga', '#SanidadPublica', 
            '#MadridSeLevantaEl13', 'sanitarios Madrid',
            'sanitarios Ayuso', 'huelga sanidad', 'ambulatorios Madrid',
            'atenci�n primaria Madrid', 'marea blanca Madrid',
            'Ayuso sanidad', 'recortes sanidad PP', 'sanidad madrile�a')

topic10 <- c('delito sedici�n', 'independentismo', 'l�deres independentistas',
             'reforma sedici�n', 'castellano aulas Catalu�a',
             'Catalu�a castellano', 'Catalu�a referendum', 'sedici�n 1-O',
             'Catalu�a 2017', 'Puigdemont', 'Junqueras', 'delito malversaci�n',
             'reforma malversaci�n')


topics = list(topic1,topic2,topic3,topic4,topic5,
              topic6,topic7,topic8,topic9,topic10)

ntweets = c(800,750,500,750,3500,
            2500,1000,2000,750,1000)

for (i in 1:length(topics)) {
  tweets <- search_tweets2(q = topics[[i]], 
                           n = ntweets[i],
                           lang = "es",
                           result_type = 'popular',
                           include_rts = FALSE,
                           retryonratelimit = TRUE)
  tweets <- tweets[,c(4)]
  tweets <- distinct(tweets)
  
  path <- paste("data/processed/1_TWEETS_RAW_BY_CLUSTER/tweets_",as.character(i),".csv",sep="")
  
  write.csv2(tweets,file=path) 
  
}





