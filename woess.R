#### Nuevas Funciones mas Potentes

 

optimal_binning <- function(df,id,x,p=0.05) {

 

  

  ctree=ctree(formula(paste(id,"~",x)),

              data=df,

              na.action=na.exclude,

              control=ctree_control(minbucket=ceiling(round(p*nrow(df)))))

 

  n=length(ctree)

 

  b=NULL

  for (i in 1:n) {

    a=ctree[i]$node$split$breaks

    b =cbind(b,a)

  }

 

  cuts<-sort(b)

  cuts<-c(-Inf,cuts,Inf)

  return(cuts)

 

}

 

 

IV_woe <- function(id,base,variable,dir,type='optimal',corte=10,p=0.05){

 

  

  if(type=='percentil') {

   

    breaks= unique(quantile(base[,variable], prob = c(seq(0, 1, by = 1/corte)),

                            type = 5,na.rm=TRUE))

    breaks[1]=-Inf

    breaks[length(breaks)]=Inf

  }

 

  

  if (type=='optimal') {

   

    breaks <- optimal_binning(base,id,variable,p)

   

  }

 

  if(type=='expert') {

   

    breaks<-corte

  }

  if (dir==1) {

   

    x1 <-  base[base[id]==1,]

    x2 <-  base[base[id]==0,]

   

    #x1 <-  base %>% filter(nIndSob==1)

    #x2 <-  base %>% filter(nIndSob==0)

    # x1 <- base %>% filter(id==1)

    #x2 <- base %>% filter(id==0)

  }

  else {

    x1 <-  base[base[id]==0,]

    x2 <-  base[base[id]==1,]

    #x1 <- filter(base, base$id==0)

    #x2 <- filter(base, base$id==1)

  }

  y1 <- cut(x1[,variable],breaks)

  y2 <- cut(x2[,variable],breaks)

  y1 <- addNA(y1)

  y2 <- addNA(y2)

  # Con 1

  a <- table(y1)

  df1 <- data.frame(a)

  rf1 <-df1$Freq/sum(df1$Freq)

  df1$relFreq <- rf1

  df1

  #Con 0

  b <- table(y2)

  df2 <- data.frame(b)

  rf2 <-df2$Freq/sum(df2$Freq)

  df1$Freq2 <- b

  df1$relFreq2 <- rf2

  #

  df1$WOEs <- log((df1$relFreq/df1$relFreq2))

  df1$WOEs[is.na(df1$WOEs)] <- 0

  df1$IV <- (df1$relFreq-df1$relFreq2)*df1$WOEs

  df1$IV[is.na(df1$IV)] <- 0

  # Puntaje

 

  de<-data.frame("Total",sum(df1$Freq),sum(df1$relFreq),sum(df1$Freq2),sum(df1$relFreq2),"",sum(df1$IV))

  names(de)<-c("y1","Freq","relFreq","Freq2","relFreq2","WOEs","IV")

  newdf <- rbind(df1, de)

 

  a<-newdf$WOEs

  dat1<-a[-length(a)]

  dat2<-dat1[-length(dat1)]

 

  bin<-df1$Freq+df1$Freq2

  bin_total<-sum(bin)

  bin_perc <- bin/bin_total

  newdf$bin_perc <-c(bin_perc,1.00)

 

  #  if (dir==1){

 

  bad_rate <- newdf$Freq2/(newdf$Freq+newdf$Freq2)

  newdf$bad_rate <-bad_rate

 

  #}

 

  #else {

  # bad_rate <- newdf$Freq/(newdf$Freq+newdf$Freq2)

  #newdf$bad_rate <-bad_rate

  #}

 

  if( all(diff(as.numeric(as.character(dat2)))>0) | all(diff(as.numeric(as.character(dat2)))<0)) {

    newdf$Type <- 'Monoticity'

  }

 

  else {

    newdf$Type <- 'Non Monoticity'

  }

 

  return(list(newdf,breaks,variable))

}

 

 

# Version 2 Final (Funciona)

massive_IV<- function(id,df,type='optimal_percentil',p=0.05,salto_pd=0.01) {

  b <- NULL 

  lista <-NULL

 

  for (i in 1:ncol(df)) {

    if (names(df[i]) != id &

        (class(df[, i]) == 'integer' | class(df[, i]) == 'numeric') & type != 'optimal_percentil')

    {

      woe <-  IV_woe(id, df, names(df[i]), 1, type, corte = 10, p)

      woe <- as.data.frame(woe[[1]])

      iv <- woe[nrow(woe), 7]

      iv <- as.data.frame(iv)

      rownames(iv)[1] <- names(df[i])

     

      

      corte_opt <- c(NA)

      corte_opt <- as.data.frame(corte_opt)

      rownames(corte_opt)[1] <- names(df[i])

      b <- rbind(b,cbind(iv,corte_opt))

     

    }

   

    else if (names(df[i]) != id &

             (class(df[, i]) == 'integer' | class(df[, i]) == 'numeric') & type == 'optimal_percentil')

    {

     

      for(j in 10:1) {

       

        

        woe<-IV_woe(id,df,

                    names(df[i]),0,type='percentil'

                    ,corte=j)

        

        

        woe<-woe[[1]]

        iv <- woe[nrow(woe), 7]

        iv <- as.data.frame(iv)

       

        

        pds<-as.numeric(woe[1:(nrow(woe)-2),9])

       

        mono<-woe[nrow(woe),10]

       

        

        #

        response <- all(diff(as.numeric(as.character(pds)))>= salto_pd) |

          all(diff(as.numeric(as.character(pds)))<= -salto_pd )

       

        #

       

        lista<-rbind(lista,cbind(j,iv,mono,response))

       

      }

      lista <- lista[lista['response']=="TRUE",]

     

      

      lista <- lista[lista['mono']=="Monoticity",]

     

      lista <- lista[lista['iv']==max(lista['iv']),]

     

      

      corte_opt<-lista$j

      corte_opt <-min(corte_opt)

     

      if (is.na(corte_opt)) {

        corte_opt=2

      }

     

      woe<-IV_woe(id,df,

                  names(df[i]),0,type='percentil'

                  ,corte=corte_opt)

      

      

      

      woe <- as.data.frame(woe[[1]])

     

      iv <- woe[nrow(woe), 7]

      iv <- as.data.frame(iv)

      rownames(iv)[1] <- names(df[i])

     

      corte_opt <- as.data.frame(corte_opt)

      rownames(corte_opt)[1] <- names(df[i])

      b <- rbind(b,cbind(iv,corte_opt))

     

    }

    else

    {

      iv <- c(NA)

      iv <- as.data.frame(iv)

      rownames(iv)[1] <- names(df[i])

     

      

      corte_opt <- c(NA)

      corte_opt <- as.data.frame(corte_opt)

      rownames(corte_opt)[1] <- names(df[i])

      b <- rbind(b,cbind(iv,corte_opt))

     

    }

    lista <- NULL

    corte_opt <- NULL

  }

  return(b)

}

 

 

IV_woe_deploy <- function(binbin,base2,mergeNA=0) {

 

  

  a<- binbin[[1]]

 

  breaks <- binbin[[2]]

 

  variable <- binbin[[3]]

 

  if (mergeNA==0){

   

    

    

    a<-a$WOEs

   

    dat1<-a[-length(a)]

    dat2<-dat1[-length(dat1)]

   

    

    vector <- cut(as.matrix(base2[variable]),breaks

                  ,labels=dat2)

   

    

    levels(vector)<-c(levels(vector),dat1[length(dat1)])  #Add the extra level to your factor

    vector[is.na(vector)] <-   dat1[length(dat1)]

   

    

    #vector <- as.data.frame(vector)

    vector <- as.numeric(as.character(vector))

   

    

    #base[base_deploy] <- vector

   

  }

 

  else  {

   

    a<-a$WOEs

   

    dat1<-a[-length(a)]

   

    row <- binbin[[4]]

   

    vector <- cut(as.matrix(base2[variable]),breaks

                  ,labels=dat1)

   

    

    

    #vector<-addNA(vector)  #Add NA

    levels(vector)<-c(levels(vector),NA)

   

    vector[is.na(vector)] <-    as.character(binbin[[1]][row,6])

   

    vector <- as.numeric(as.character(vector))

  }

 

  # training_data$END_ENT_SALDO_TOTAL_UM_binning_NA[is.na(training_data$END_ENT_SALDO_TOTAL_UM_binning_NA)]<-

  #    bin_END_POR_UTI_LINEA_UM_NA[[1]][2,6]

 

  

  return( vector)

}

 

 

merge_bin_na <- function(binbin,row) {

 

  bin_prueba<-binbin

 

  col1<-data.frame(lapply(binbin[[1]][1], as.character),stringsAsFactors = FALSE)

 

  a<-bin_prueba[[1]][,2:ncol(bin_prueba[[1]])]

 

  a<- cbind(col1,a)

 

  Freq<-a[nrow(a)-1,2]+a[row,2]

  Freq2<-a[nrow(a)-1,4]+a[row,4]

 

  d<- as.data.frame(

    cbind(

      paste0(a[row,1],' & NA'),

      as.numeric(Freq),

      as.numeric(Freq/a[nrow(a),2]),

      as.numeric(Freq2),

      as.numeric(Freq2/a[nrow(a),4]))

    ,stringsAsFactors = FALSE)

 

  

  

  names(d)<-c("y1","Freq","relFreq","Freq2","relFreq2")

 

  a$WOEs<-NULL

  a$IV<-NULL

  a$bin_perc<-NULL

  a$bad_rate<-NULL

  a$Type<-NULL

 

  a[row,1]<-as.character(d$y1)

  a[row,2]<-as.numeric(d$Freq)

  a[row,3]<-as.numeric(d$relFreq)

  a[row,4]<-as.numeric(d$Freq2)

  a[row,5]<-as.numeric(d$relFreq2)

  df1<-a[1:(nrow(a)-2),]

 

  ##

 

  df1$WOEs <- log((df1$relFreq/df1$relFreq2))

  df1$WOEs[is.na(df1$WOEs)] <- 0

  df1$IV <- (df1$relFreq-df1$relFreq2)*df1$WOEs

  df1$IV[is.na(df1$IV)] <- 0

 

  

  de<-data.frame("Total",sum(df1$Freq),sum(df1$relFreq),sum(df1$Freq2),sum(df1$relFreq2),"",sum(df1$IV))

  names(de)<-c("y1","Freq","relFreq","Freq2","relFreq2","WOEs","IV")

  newdf <- rbind(df1, de)

 

  w<-newdf$WOEs

  dat1<-w[-length(w)]

  dat2<-dat1[-length(dat1)]

 

  bin<-df1$Freq+df1$Freq2

  bin_total<-sum(bin)

  bin_perc <- bin/bin_total

  newdf$bin_perc <-c(bin_perc,1.00)

 

  

  bad_rate <- newdf$Freq2/(newdf$Freq+newdf$Freq2)

  newdf$bad_rate <-bad_rate

 

  

  if( all(diff(as.numeric(as.character(dat2)))>0) | all(diff(as.numeric(as.character(dat2)))<0)) {

    newdf$Type <- 'Monoticity'

  } else {

   newdf$Type <- 'Non Monoticity' }

 

  

  

  list(

    newdf,

    binbin[[2]],

    binbin[[3]],

    row

  )

 

  

}

 

 

scoreCard <- function(binbin,model) {

  # Los objetos deben acabar en _binning

  a<-cbind(binbin[[3]],binbin[[1]])

  colnames(a)[1] <- 'Variable'

  colnames(a)[2] <- 'Rango'

  colnames(a)[3] <- 'N_Buenos'

  colnames(a)[4] <- '%Buenos'

  colnames(a)[5] <- 'N_Malos'

  colnames(a)[6] <- '%Malos'

  beta<-model$coefficients[paste0(binbin[[3]],'_binning')]

  if (is.na(beta)) {

    beta<-model$coefficients[paste0(binbin[[3]],'_NA_binning')]

  }

  else {

    beta<-beta

  }

  a$WOEs <- as.numeric(a$WOEs)*100

  a$beta<-beta

  a$puntaje <- a$beta*as.numeric(a$WOEs)

  a$puntaje_aj <- 80/log(2)*a$puntaje/100+model$coefficients['(Intercept)']/(length(model$coefficients)-1)*80/log(2)

  a<- a[1:nrow(a)-1,]

  return(a)

}

 

 

decilimizar <- function(variable,type='decil',corte=10) {

 

  

  if(type=='decil') {

    x_breaks <-  unique(quantile(as.numeric(unlist(variable)),

                                 prob = c(seq(0, 1, by = 1/corte)),

                                 #                  prob = c(-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1),

                                 type = 5,na.rm=TRUE))

   

    x_breaks[1]=-Inf

    x_breaks[length(x_breaks)]=Inf

   

    y1 <- cut(as.numeric(unlist(variable)),x_breaks)

   

    return(y1)

  }

 

  else if(type=='expert') {

    x_breaks <-  corte

   

    y1 <- cut(as.numeric(unlist(variable)),x_breaks)

   

    return(y1)

   

  }

 

}
