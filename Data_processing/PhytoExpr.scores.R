library(readr)
setwd('/Volumes/N1')

bdi.pred <- read_csv('./PhytoExpr/bdi.newexppred.sequences.csv')
bdi.pred <- bdi.pred[which(!is.na(bdi.pred$model1) & !is.na(bdi.pred$model2) & !is.na(bdi.pred$model3) & !is.na(bdi.pred$model4) & !is.na(bdi.pred$model5) ),]

bdi.ref <- bdi.pred[which(is.na(bdi.pred$locus)),]
bdi.alt <- bdi.pred[which(!is.na(bdi.pred$locus)),]

bdi.ref <- bdi.ref[c(4,18,19,20,21,22)]
bdi.alt <- bdi.alt[c(3:22)]

bdi.ref<-bdi.ref[which(!is.na(bdi.ref$model1)),]
bdi.alt<-bdi.alt[which(!is.na(bdi.alt$model1)),]

bdi.diff<-merge(bdi.ref, bdi.alt, by='transcript', all=T)

bdi.diff$score.ref<-((bdi.diff$model1.x+bdi.diff$model2.x+bdi.diff$model3.x+bdi.diff$model4.x+bdi.diff$model5.x)/5)
bdi.diff$score.alt<-((bdi.diff$model1.y+bdi.diff$model2.y+bdi.diff$model3.y+bdi.diff$model4.y+bdi.diff$model5.y)/5)
bdi.diff$variant <- paste(bdi.diff$chromosome,bdi.diff$pos,bdi.diff$ref,bdi.diff$alt,sep=':')

bdi.diff<-bdi.diff[c(1,7:20,26:28)]
bdi.diff<-bdi.diff[which(!is.na(bdi.diff$score.ref) & !is.na(bdi.diff$score.alt)),]

bdi.alt<-with(bdi.diff[c(18,17)], aggregate(score.alt ~ variant, FUN = mean ))
bdi.ref<-with(bdi.diff[c(18,16)], aggregate(score.ref ~ variant, FUN = mean ))

write.table(merge(bdi.ref,bdi.alt, by='variant'),file='./PhytoExpr/scores.ref.alt.tsv', sep='\t', quote = F, row.names = F, col.names = T)

bdi.diff$score <- bdi.diff$score.alt - bdi.diff$score.ref
nrow(bdi.diff)
bdi.diff<-with(bdi.diff[c(18:19)], aggregate(score ~ variant, FUN = mean ))
nrow(bdi.diff)
write.table(bdi.diff,file='./PhytoExpr/scores.tsv', sep='\t', quote = F, row.names = F, col.names = T)
