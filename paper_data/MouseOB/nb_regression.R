if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("sva")
#library(sva)
library(MASS)

counts <- read.table(file='/Users/user/GPcounts/data/MouseOB/Rep11_MOB_0.csv', header=TRUE, sep=',')
cells <- read.table(file='/Users/user/GPcounts/data/MouseOB/locations.csv', header=TRUE, sep=',')

rownames(counts) = counts[,1]
rownames(cells) = cells[,1]
counts = counts[,-1]
cells = cells[,-1]

counts = cbind(counts,cells$total_counts)
colnames(counts)[14860] = 'total_counts'
coeffs_nb <-list()
scales_nb <- list()
total_counts = c(cells$total_counts)

for (x in c(1:100))
{
  results<-glm.nb(formula=counts[,x]~0+counts$total_counts,link=identity, data=counts)
  coeffs = as.data.frame(results$coefficients)
  scales <- results$coefficients*total_counts
  scales = as.data.frame(scales)
  scales_nb <- append(scales_nb,scales)
}

scales_nb=write.table(scales_nb,sep = "\t","../data/MouseOB/scales_nb.txt",col.names = TRUE, row.names = F)