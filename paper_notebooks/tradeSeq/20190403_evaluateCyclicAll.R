setwd("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/")
library(slingshot)
library(RColorBrewer)
library(mgcv)
library(tradeSeq)
library(edgeR)
library(rafalib)
library(wesanderson)
library(BiocParallel)
library(doParallel)
NCORES <- 2

  FQnorm <- function(counts){
    rk <- apply(counts,2,rank,ties.method='min')
    counts.sort <- apply(counts,2,sort)
    refdist <- apply(counts.sort,1,median)
    norm <- apply(rk,2,function(r){ refdist[r] })
    rownames(norm) <- rownames(counts)
    return(norm)
  }

dataAll <- readRDS("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/datasets_for_koen.rds")
  

for(datasetIter in 1:10){
 
  pdf(paste0("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/dataset",datasetIter,".pdf"))

  data <- dataAll[[datasetIter]]
  counts <- as.matrix(t(data$counts))
  falseGenes <- data$feature_info$gene_id[!data$feature_info$housekeeping]
  nullGenes <- data$feature_info$gene_id[data$feature_info$housekeeping]

  pal <- wes_palette("Zissou1", 12, type = "continuous")
  truePseudotime <- data$prior_information$timecourse_continuous
  g <- Hmisc::cut2(truePseudotime,g=12)

  # quantile normalization
  normCounts <- FQnorm(counts)

  ## dim red
  pca <- prcomp(log1p(t(normCounts)), scale. = FALSE)
  rd <- pca$x[,1:2]
  plot(rd, pch=16, asp = 1, col=pal[g])

  library(princurve)
  pcc <- principal_curve(rd, smoother="periodic_lowess")
  lines(x=pcc$s[order(pcc$lambda),1], y=pcc$s[order(pcc$lambda),2], col="red", lwd=2)

  # fit smoothers on raw data
  cWeights <- rep(1,ncol(counts))
  pseudoT <- matrix(pcc$lambda,nrow=ncol(counts),ncol=1)
  gamList <- fitGAM(counts, pseudotime=pseudoT, cellWeights=cWeights, nknots=5)
  
  write.csv(pseudoT, file = paste0("pseudoT_Index_",datasetIter,".csv"))
  
  # Test for association of expression with the trajectory
  assocTestRes <- associationTest(gamList)
 
  # ### tradeSeq on true pseudotime
  cWeights <- rep(1,ncol(counts))
  pst <- matrix(truePseudotime, nrow=ncol(counts), ncol=1, byrow=FALSE)
  gamListTrueTime <- fitGAM(counts, pseudotime=pst, cellWeights=cWeights)
  assocTestTrueRes <- associationTest(gamListTrueTime)

  ### tradeSeq with 3 knots (Monocle default)
  gamListTrueTime_3k <- fitGAM(counts, pseudotime=pst, cellWeights=cWeights, nknots=3)
  assocTestTrueRes_3k <- associationTest(gamListTrueTime_3k)

  ### tradeSeq with 5 knots (optimal acc. to AIC)
  gamListTrueTime_5k <- fitGAM(counts, pseudotime=pst, cellWeights=cWeights, nknots=5)
  assocTestTrueRes_5k <- associationTest(gamListTrueTime_5k)

  ########
  # FDP-TPR
  ########
  library(iCOBRA)
  library(scales)
  truth <- as.data.frame(matrix(rep(0,nrow(counts)), dimnames=list(rownames(counts),"status")))
  truth[falseGenes,"status"] <- 1
  #write.csv(truth, file = paste0("truth",datasetIter,".csv"))
  #write.csv(assocTestRes$pval, file = paste0("pval_cyclic",datasetIter,".csv"))
  
  GP_NB_counts <- read.csv(file=paste0("ll_Negative_binomial_Cyclic_Index_", datasetIter, ".csv"), header=TRUE,sep=",")
  GP_G_counts <- read.csv(file=paste0("ll_Gaussian_Cyclic_Index_", datasetIter, ".csv"), header=TRUE,sep=",")
  GP_real_NB_counts <- read.csv(file=paste0("ll_true_Negative_binomial_Cyclic_Index_", datasetIter, ".csv"), header=TRUE,sep=",")
  GP_real_G_counts <- read.csv(file=paste0("ll_true_Gaussian_Cyclic_Index_", datasetIter, ".csv"), header=TRUE,sep=",")
  
  ### estimated pseudotime
  pval <- data.frame( #tradeSeq_slingshot_assoc=assocTestRes$pval,
                      tradeSeq_true_10k=assocTestTrueRes$pval,
                      tradeSeq_true_3k=assocTestTrueRes_3k$pval,
                      tradeSeq_true_5k=assocTestTrueRes_5k$pval,
                        row.names=rownames(counts))
  score <- data.frame(
    #GPcounts_NB=GP_NB_counts$log_likelihood_ratio,
    #GPcounts_Gaussian=GP_G_counts$log_likelihood_ratio,
    GPcounts_true_NB=GP_real_NB_counts$log_likelihood_ratio,
    GPcounts_true_Gaussian=GP_real_G_counts$log_likelihood_ratio,
    row.names=rownames(counts))
  
  cobra <- COBRAData(pval=pval, truth=truth,score=score)
  saveRDS(cobra, file=paste0("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/cobra",datasetIter,".rds"))
  dev.off()
}
