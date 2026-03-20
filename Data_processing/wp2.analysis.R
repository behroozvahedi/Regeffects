library(data.table)
library(ggfortify)
library(ggplot2)
library(viridis)
library(MASS)

#set this to the path of the external drive with all the necessary data
setwd('/Volumes/N1/Embeddings')

get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  return(dens$z[ii])
}

#---------------------------------- PEER ---------------------------------------

# number of factors
n.factors<-10

#residuals <- read.csv(paste('./PEER/peer_n-',n,'/residuals.csv', sep=''), header=F)
factors <- read.csv(paste('./DATA/PEER/peer_n-',n.factors,'/X.csv', sep=''), header=F)

#genes<-read.table('./DATA/peer.genes.csv',header=F,sep='\t', stringsAsFactors = F)
samples<-read.table('./DATA/peer.samples.csv',header=F,sep='\t', stringsAsFactors = F)

#names(residuals)<-samples$V1
#rownames(residuals)<-genes$V1

df.peer = data.frame('id'= samples$V1, t(factors), stringsAsFactors = F)

# plot the first PEER factor against the second factor
plot(df.peer$X1,df.peer$X2)

#------------------------------- PCA -------------------------------------------
expression <- read.csv('./DATA/peer.expression.csv',header = F)
genes <- read.csv('./DATA/peer.genes.csv',header = F)$V1
samples <- read.csv('./DATA/peer.samples.csv',header = F)$V1

colnames(expression)<-genes
rownames(expression)<-samples

pca<-prcomp(expression[, colSums(expression) != 0, drop = FALSE], scale. = TRUE)
autoplot(pca)

# Scree plot
eigenvalues <- pca$sdev^2
plot(eigenvalues/sum(eigenvalues), type = "b",
     xlab = "Principal Component",
     ylab = "Percentage of Variance Explained")
#abline(v = 20, col = "red")

df.pca <- data.frame('id' = names(pca$x[,1]),pca$x[,c(1:20)],stringsAsFactors = F)

# Scatter plot PCA1 / PCA2
plot(df.pca$PC1,df.pca$PC2)

#----------------------- PCA/PEER correlation ----------------------------------

pca.peer<-merge(df.pca,df.peer,by='id')
#Check correlation between PCA1 and 1st PEER factor
cor(pca.peer$PC1,pca.peer$X1)
cor.test(pca.peer$PC1,pca.peer$X1)

#Check correlation between PCA2 and 2nd PEER factor
cor(pca.peer$PC2,pca.peer$X2)
cor.test(pca.peer$PC2,pca.peer$X2)

#------------------------------- ANALYSIS --------------------------------------
# Check differences between C and M
# Assess correlation between observed and predicted for average expression among C (y_mean)
# Assess correlation between observed and predicted for deviation from y_mean among M

# Which model's predictions to analyse, 'none': only caduceus embeddings, 'pred': caduceus embeddings+a2z prediction, 'emb': caduceus+a2z embeddings 
for (type in c('none','pred', 'a2z', 'emb'))
{
  # The dataset with predicted and RNAseq read TPM values for each gene
  data <- read.table(paste('./DATA/wp2.dataset_',type,'.tsv',sep=''), header=T, sep='\t',fill=T,colClasses = c('character','character','character','character','integer','double','double','double','double','double','double'),stringsAsFactors = F)
  
  # List of primary transcripts
  primary <- read.table('./DATA/BdistachyonBd21_3_537_v1.2.protein_primaryTranscriptOnly.tsv', header=T, sep='\t',fill=T,stringsAsFactors = F)
  
  # Keep only primary transcript predictions
  data<-data[which(data$transcript %in% primary$transcript),]
  # Average predictions from the 5 models into one value
  data$prediction <- (data$model_1_pred+data$model_2_pred+data$model_3_pred+data$model_4_pred+data$model_5_pred)/5
  # Log transform the RNAseq TPM reads
  data$tpm <-log10(1+data$tpm)
  
  length(unique(data$gene))
  length(unique(data$transcript))
  length(unique(data$hash.seq.))/nrow(data)
  
  data<-data[c(1,2,4,5,11,12)]
  
  #Get RNAseq line to plant id translations
  id2line <- read.table('./DATA/RNAseq/RNAseq.ids.tsv', header=T, sep='\t',fill=T,stringsAsFactors = F)
  
  
  # Label plants according to their RNAseq batch
  batch1<-c(201,202,203,204,205,206,207,208,210,219,220,221,222,223,225,226,227,228,229,230,231,232,233,234,235,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,279,280,282,283,284,285,286,287,288,289,290,291,292,293,294,296,297,298,299,300,301,302,303,304,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,376,377,378,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,399,400)
  batch2<-c(401,402,403,404,405,406,407,408,409,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,434,435,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,475,476,477,478,480,481,482,483,485,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,576,577,578,580,581,582,583,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600)
  batch3<-c(1,2,3,4,5,6,7,8,9,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,32,33,34,35,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,69,70,71,72,73,74,76,77,79,80,81,82,83,84,85,86,87,88,89,90,91,92,94,95,96,97,98,99,151,152,153,154,155,157,158,159,160,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,185,186,187,188,189,190,191,192,193,194,196,197,198,199,200,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857)
  batch4<-c(601,602,603,604,605,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,752,753,754,755,756,758,759,760,763,764,765,766,767,770,771,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800)
  data$batch <- NA
  data[which(data$id %in% id2line[which(id2line$ID %in% batch1),]$line),]$batch = 'batch1'
  data[which(data$id %in% id2line[which(id2line$ID %in% batch2),]$line),]$batch = 'batch2'
  data[which(data$id %in% id2line[which(id2line$ID %in% batch3),]$line),]$batch = 'batch3'
  data[which(data$id %in% id2line[which(id2line$ID %in% batch4),]$line),]$batch = 'batch4'

  #Get RNAseq QC data and filter out plants that do not meet RNAseq QC limit
  QC.rnaseq <- rbind(read.table("./DATA/RNAseq/90-1106610255/01_output/QC.tsv", header=T, sep='\t',fill=T,check.names = F,stringsAsFactors = F),read.table("./DATA/RNAseq/90-1120364334/01_output/QC.tsv", header=T, sep='\t',fill=T,check.names = F,stringsAsFactors = F),read.table("./DATA/RNAseq/90-1147085216/01_output/QC.tsv", header=T, sep='\t',fill=T,check.names = F,stringsAsFactors = F),read.table("./DATA/RNAseq/90-1080311147/01_output/QC.tsv", header=T, sep='\t',fill=T,check.names = F,stringsAsFactors = F))
  QC.filtered <- id2line[which(id2line$ID %in% QC.rnaseq[which(QC.rnaseq$`Aligned concordantly 1 time` < 70),]$line),]$line
  data<-data[which(!(data$id %in% QC.filtered)),]
  
  # Identify controls
  data$CONTROL <- startsWith(data$id,'C')
  
  # Average consolidate all controls into one averaged dataset
  controls <- data[which(data$CONTROL == TRUE),]
  controls<-aggregate(controls[,5:6], list('gene'=controls$gene), mean)
  
  # seperate out the cases
  cases <- data[which(data$CONTROL == FALSE),]
  
  # Assess correlation between observed and predicted for average expression among meaned controls
  cor(controls$tpm, controls$prediction)
  
  df.cor<-data.frame('group'='controls','correlation'=  cor(controls$tpm, controls$prediction),'p.value'=cor.test(controls$tpm,controls$prediction)$p.value, stringsAsFactors = F)
  
  cases$dev_prediction <- NaN
  cases$dev_tpm <- NaN
  dt_controls = data.table(controls)
  dt_cases = data.table(cases)
  
  # Calculate how much each case deviates from controls for each gene
  for (gene_x in unique(cases$gene))
  {
    dt_cases[gene == gene_x, dev_prediction := prediction - dt_controls[gene == gene_x, prediction]]
    dt_cases[gene == gene_x, dev_tpm := tpm - dt_controls[gene == gene_x, tpm]]
  }
  
  # remove all rows where the predicted value for control equals the case as they likely have the same sequence (in short both are wild type)
  cases.unique <- dt_cases[which(dt_cases$dev_prediction != 0),]
  
  #plot predicted versus measured tpms
  plot <- ggplot(cases.unique, aes(x=dev_prediction, y=dev_tpm)) + geom_point() + geom_smooth(method = "lm") + ggtitle('Deviations from controls') + xlab('Predicted') + ylab('Observed')
  ggsave(
    paste("./RESULTS/results.plot1.",type,".png",sep=''),
    plot = plot,
    device = 'png',
    scale = 1,
    width = 10,
    height = 10,
    units = "in",
    dpi = 300,
    limitsize = TRUE,
    bg = NULL,
  )

  #measure correlation between predicted and measured TPM
  df.cor<-rbind(df.cor,data.frame('group'='cases (deviation)','correlation'=  cor(cases.unique$dev_prediction,cases.unique$dev_tpm),'p.value'=cor.test(cases.unique$dev_prediction,cases.unique$dev_tpm)$p.value, stringsAsFactors = F))
  write.table(df.cor,file=paste('./RESULTS/results.correlation.',type,'.tsv',sep=''), sep='\t', quote = F, row.names = F)
  
  #Try the same for only in cases there the absolute deviation is above 0.01
  tmp <- dt_cases[which(abs(dt_cases$dev_prediction) > 0.01),]
  plot(tmp$dev_prediction,tmp$dev_tpm)
  cor(tmp$dev_prediction,tmp$dev_tpm)
  cor.test(tmp$dev_prediction,tmp$dev_tpm)
  
  # Make linear model controlled for batch
  summary(lm(dev_tpm ~ dev_prediction + batch, data=cases.unique))
  
  # Make linear model adjusted for PCA components
  df <- merge(cases.unique, df.pca, by='id')
  #df<-data.frame(df)[,c(9:ncol(df))] #select all components
  df<-data.frame(df)[,c(9:25)] #select the first 15 components
  summary(lm(dev_tpm ~ . , data=df))
  
  # Make linear model controlled for PEER factors
  df <- merge(cases.unique, df.peer, by='id')
  df<-data.frame(df)[,c(9:ncol(df))] #select all factors
  summary(lm(dev_tpm ~ . , data=df))
  
  library(flexmix)
  
  # plot BIC for models with increasing number of PEER factors included
  #df.BIC <- data.frame('factors'=0,'BIC'=BIC(lm(dev_tpm ~ dev_prediction , data=cases.unique)))
  df.BIC <- data.frame(stringsAsFactors = F)
  for (n.factors in 1:n.factors )
  {
    print(n.factors)
    tmp <- read.csv(paste('./DATA/PEER/peer_n-',n.factors,'/X.csv', sep=''), header=F)
    samples<-read.table('./DATA/peer.samples.csv',header=F,sep='\t', stringsAsFactors = F)
    df = data.frame('id'= samples$V1, t(tmp), stringsAsFactors = F)
    df <- merge(cases.unique, df, by='id')
    df<-data.frame(df)[,c(10:ncol(df))]
    df.BIC <- rbind(df.BIC, data.frame('factors'=n.factors,'BIC'=BIC(lm(dev_tpm ~ . , data=df))))
    
  }
  
  plot <- ggplot(df.BIC, aes(x=factors, y=BIC)) + geom_point(size=3, shape=1) + geom_line() + ggtitle('BIC plot') + xlab('PEER Factors') + ylab('BIC')
  ggsave(
    paste("./RESULTS/results.plot2.",type,".png",sep=''),
    plot = plot,
    device = 'png',
    scale = 1,
    width = 10,
    height = 10,
    units = "in",
    dpi = 300,
    limitsize = TRUE,
    bg = NULL,
  )
  
  #plot(df.BIC, type = "b",
  #     xlab = "PEER factors",
  #     ylab = "BIC")
  
  # plot BIC for models with increasing number of PCAs included
  #df.BIC <- data.frame('factors'=0,'BIC'=BIC(lm(dev_tpm ~ dev_prediction , data=cases.unique)))
  df.BIC <- data.frame(stringsAsFactors = F)
  for (n.pca in 1:25 )
  {
    print(n.pca)
    df <- data.frame('id' = names(pca$x[,1]),pca$x[,c(1:n.pca)],stringsAsFactors = F)
    df <- merge(cases.unique, df, by='id')
    df<-data.frame(df)[,c(10:(10+n.pca))]
    df.BIC <- rbind(df.BIC, data.frame('factors'=n.pca,'BIC'=BIC(lm(dev_tpm ~ . , data=df))))
  }
  
  plot <- ggplot(df.BIC, aes(x=factors, y=BIC)) + geom_point(size=3, shape=1) + geom_line() + ggtitle('BIC plot') + xlab('PCAs') + ylab('BIC')
  ggsave(
    paste("./RESULTS/results.plot3.",type,".png",sep=''),
    plot = plot,
    device = 'png',
    scale = 1,
    width = 10,
    height = 10,
    units = "in",
    dpi = 300,
    limitsize = TRUE,
    bg = NULL,
  )
  
  #plot(df.BIC, type = "b",
  #     xlab = "PCAs included",
  #     ylab = "BIC")
  
  #........................... FINAL MODELS ......................................
  
  results <- list()
  df.results <- data.frame(stringsAsFactors = F)
  
  # Make linear model adjusted for optimal number of PCA components
  pca.optimal <- 20
  df <- merge(cases.unique, df.pca, by='id')
  df<-data.frame(df)[,c(9:(10+pca.optimal))] #select the optimal number of components
  res<-lm(dev_tpm ~ . , data=df)
  summary(res)
  results['PCA'] = list(res)
  df.results <- rbind(df.results,data.frame('adjustment'=paste('PCA-',pca.optimal,sep=''), 'coef'=coef(summary(res))[2,1],'p'=coef(summary(res))[2,4],stringsAsFactors = F))
  
  # Make linear model controlled for optimal number of PEER factors
  peer.optimal <- 10
  tmp <- read.csv(paste('./DATA/PEER/peer_n-',peer.optimal,'/X.csv', sep=''), header=F)
  samples<-read.table('./DATA/peer.samples.csv',header=F,sep='\t', stringsAsFactors = F)
  df.peer.optimal = data.frame('id'= samples$V1, t(tmp), stringsAsFactors = F)
  df <- merge(cases.unique, df.peer.optimal, by='id')
  df<-data.frame(df)[,c(9:ncol(df))]
  res <- lm(dev_tpm ~ . , data=df)
  summary(res)
  results['PEER'] = list(res)
  df.results <- rbind(df.results,data.frame('adjustment'=paste('PEER-',peer.optimal,sep=''), 'coef'=coef(summary(res))[2,1],'p'=coef(summary(res))[2,4],stringsAsFactors = F))
  
  saveRDS(results,file=paste('./RESULTS/results.model.',type,'.rds',sep=''))
  write.table(df.results,file=paste('./RESULTS/results.model.',type,'.tsv',sep=''), sep='\t', quote = F, row.names = F)
}

#-------------------------------------END---------------------------------------

#data[which(data$CONTROL == T),] = 

cor(dt_cases$dev_prediction,dt_cases$dev_tpm)
hist(dt_cases$dev_prediction)
hist(dt_cases$dev_tpm)

cor(dt_cases$prediction,dt_cases$tpm)

#---- PEER ---
n<-15
residuals <- read.csv(paste('/Volumes/N1/PEER/peer_n-',n,'/residuals.csv', sep=''), header=F)
factors <- read.csv(paste('/Volumes/N1/PEER/peer_n-',n,'/X.csv', sep=''), header=F)
#weights <- read.csv(paste('/Volumes/N1/PEER/peer_n-',n,'/W.csv', sep=''), header=F)

genes<-read.table('/Volumes/N1/DATA/peer.genes.csv',header=F,sep='\t', stringsAsFactors = F)
samples<-read.table('/Volumes/N1/DATA/peer.samples.csv',header=F,sep='\t', stringsAsFactors = F)

names(residuals)<-samples$V1
rownames(residuals)<-genes$V1
#names(factors)<-samples$V1
factors = data.frame('id'= samples$V1, t(factors), stringsAsFactors = F)

merge()

#---- ------

expression <- read.csv('/Volumes/N1/DATA/peer.expression.csv',header = F)
genes <- read.csv('/Volumes/N1/DATA/peer.genes.csv',header = F)$V1
samples <- read.csv('/Volumes/N1/DATA/peer.samples.csv',header = F)$V1

colnames(expression)<-genes
rownames(expression)<-samples

#df<-t(df)
#df<-scale(df)

# R Code to Drop Columns with a Sum of Zero

#Here's how to remove columns from a data frame where the sum of all values in the column equals zero:

# Sample data frame
df <- data.frame(
  a = c(0, 0, 0),
  b = c(1, 2, 3),
  c = c(0, 1, -1),
  d = c(0, 0, 0)
)

# Method 1: Using colSums() and subsetting
df_filtered <- df[, colSums(df) != 0, drop = FALSE]

# Method 2: Using dplyr
library(dplyr)
df_filtered <- df %>% select(where(~ sum(.) != 0))

# Method 3: Using purrr's discard
library(purrr)
df_filtered <- df %>% discard(~ sum(.) == 0)

# Result
print(df_filtered)
#```

### Explanation:
#1. **colSums() approach**: Calculates the sum of each column and keeps only those where the sum is not zero
#2. **dplyr approach**: Uses `select()` with the `where()` helper function to select columns based on a condition
#3. **purrr approach**: Uses `discard()` to remove columns that meet the condition (sum equals zero)

#All three methods will produce the same result - a data frame with only columns that have non-zero sums.

#Note: If you have NA values in your data, you may want to add `na.rm = TRUE` to the sum calculations.

#------------------------------- PCA -------------------------------------------
pca<-prcomp(expression[, colSums(expression) != 0, drop = FALSE], scale. = TRUE)
autoplot(pca)

eigenvalues <- pca$sdev^2
plot(eigenvalues/sum(eigenvalues), type = "b",
     xlab = "Principal Component",
     ylab = "Percentage of Variance Explained")
abline(v = 2, col = "red")

df<-merge(data.frame('id' = names(pca$x[,1]),pca$x[,c(1:20)],stringsAsFactors = F),factors,by='id')

cor(df$PC1,df$X1)
cor.test(df$PC1,df$X1)

cor(df$PC2,df$X2)
cor.test(df$PC2,df$X2)

cor(df$PC3,df$X3)
cor.test(df$PC3,df$X3)

cor(df$PC4,df$X4)
cor.test(df$PC4,df$X4)


#---
prcomp(dt_cases)

library(ggfortify)
df <- iris[1:4]
pca_res <- prcomp(df, scale. = TRUE)

pca_res <- prcomp(data[,c(4,5,6)], scale. = TRUE)

df<-dt_cases[,c(1,2,5,6)]
df$id<-factor(df$id)
df$gene<-factor(df$gene)
scale(df)

autoplot(pca_res)

dt_main <- data.table(
  ID = 1:5,
  value = c("A", "D", "C", "D", "B"),
  C = c(100, 200, 300, 400, 500),
  B = NA_real_
)

dt_lookup <- data.table(
  X = c("A", "B", "C", "D"),
  Y = c(10, 20, 30, 123)  # Value for "D" is 123
)

# Efficient join and update in one step
dt_main[value == "D", B := C - dt_lookup[X == "D", Y]]

gene <- '1G0123500'

cases[which(cases$gene == gene),]$tpm-controls[which(controls$gene == gene),]$tpm
cases[which(cases$gene == gene),]$prediction-controls[which(controls$gene == gene),]$prediction

#cor(controls$tpm,contorls$prediction)

deviations <- function(gene)
{
  #print(gene)
  dev_prediction <- (cases[which(cases$gene == gene),]$prediction -controls[which(controls$gene == gene),]$prediction)
  dev_tpm <- (cases[which(cases$gene == gene),]$tpm-controls[which(controls$gene == gene),]$tpm)
  cases[which(cases$gene == gene),]$dev_prediction = dev_prediction
  cases[which(cases$gene == gene),]$dev_tpm = dev_tpm
  return()
}

cases$dev_prediction <- NA
cases$dev_tpm <- NA
lapply(c("1G0000300","1G0000700","1G0001100","1G0001200","1G0001300"), deviations)

#head(data)

#hist(log(abs(((data$model_1_pred+data$model_2_pred2+data$model_3_pred+data$model_4_pred+data$model_5_pred)/5)- data$tpm)))
#hist(log(data$tpm))
#hist(log(abs(((data$model_1_pred+data$model_2_pred2+data$model_3_pred+data$model_4_pred+data$model_5_pred)/5))))

log10(10)
log(10)
log2(10)
length(unique(data$gene))
length(unique(data$transcript))
length(unique(data$hash.seq.))

controls <- data[which(data$CONTROL == TRUE),]
controls<-aggregate(controls[,5:6], list('gene'=controls$gene), mean)

cor(controls$tpm,controls$prediction)
