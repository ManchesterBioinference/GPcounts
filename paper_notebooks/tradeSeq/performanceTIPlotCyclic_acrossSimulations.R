setwd("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/")
library(here)
library(slingshot)
library(RColorBrewer)
library(mgcv)
library(tradeSeq)
library(edgeR)
library(rafalib)
library(wesanderson)
library(ggplot2)
library(cowplot)
library(iCOBRA)
library(scales)
library(dplyr)


### prepare performance plots ####
cols <- c(rep(c("#C6DBEF", "#08306B"), each = 3), "#4292C6", "#4daf4a",
          "#e41a1c", "#e78ac3", "#ff7f00", "darkgoldenrod1")
names(cols) <- c("tradeSeq_slingshot_end", "tradeSeq_GPfates_end", "tradeSeq_3_knots",
                 "tradeSeq_slingshot_pattern", "tradeSeq_GPfates_pattern",
                 "tradeSeq_10_knots", "tradeSeq_5_knots", "GPcounts_NB",
                 "tradeSeq_true_10k", "GPcounts_true_Gaussian", "GPcounts_Gaussian", "GPcounts_true_NB")
linetypes <- c(rep(c("solid", "solid", "solid"), 2), rep("solid", 7))
names(linetypes) <- c("tradeSeq_slingshot_end", "tradeSeq_GPfates_end", "tradeSeq_3_knots",
                      "tradeSeq_slingshot_pattern", "tradeSeq_GPfates_pattern",
                      "tradeSeq_10_knots", "tradeSeq_5_knots", "GPcounts_NB",
                      "tradeSeq_true_10k", "GPcounts_true_Gaussian", "GPcounts_Gaussian", "GPcounts_true_NB")
theme_set(theme_bw())
theme_update(legend.position = "none",
             panel.grid.major.y = element_blank(),
             panel.grid.minor.y = element_blank(),
             panel.grid.major.x = element_line(linetype = "dashed", colour = "black"),
             panel.grid.minor.x = element_line(linetype = "dashed", colour = "grey"),
             axis.title.x = element_text(size = rel(1)),
             axis.title.y = element_text(size = rel(1)),
             axis.text.x = element_text(size = rel(.8)),
             axis.text.y = element_text(size = rel(.8))
             )

dir <-("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/")

cobraFiles <- list.files(dir, pattern="cobra*", full.names=TRUE)
cobraFiles <- cobraFiles[c(1,3:10,2)] #order from 1 to 10

plotPerformanceCurve <- function(cobraObject){
  cn <- colnames(pval(cobraObject))
  cn <- gsub(cn,pattern="tradeR",replacement="tradeSeq")
  
  colnames(pval(cobraObject)) <- cn
  colnames(cobraObject@pval) <- gsub(colnames(cobraObject@pval),pattern="tradeR",replacement="tradeSeq")

  cobraObject <- calculate_adjp(cobraObject)
  cobraObject <- calculate_performance(cobraObject, binary_truth = "status")

  DyntoyPlot <- data.frame(FDP = cobraObject@fdrtprcurve$FDR,
                           TPR = cobraObject@fdrtprcurve$TPR,
                           method = cobraObject@fdrtprcurve$method,
                         cutoff = cobraObject@fdrtprcurve$CUTOFF)
  pDyntoy <- ggplot(DyntoyPlot, aes(x = FDP, y = TPR, col = method)) +
    geom_path(size = 1, aes(linetype = method)) +
    xlab("FDP") +
    scale_x_continuous(limits = c(0, 1), breaks = c(0.01, 0.05, 0.1, 0.5, 1),
                       minor_breaks = c(0:5) * .1) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_color_manual(values = cols, breaks = names(cols)) +
    scale_linetype_manual(values = linetypes, breaks = names(linetypes)) + ylab("TPR")                                                                                                                      
    pDyntoy
}

for(ii in 1:length(cobraFiles)){
    cobra <- readRDS(cobraFiles[ii])
    assign(paste0("bifplot",ii),plotPerformanceCurve(cobra))
}

p1 <- plot_grid(bifplot1, bifplot2, bifplot3, bifplot4, bifplot5,
           bifplot6, bifplot7, bifplot8, bifplot9, bifplot10,
        nrow=2, ncol=5)
p1

plotsBif <- sapply(cobraFiles, function(file){
  cobra <- readRDS(file)
  plotPerformanceCurve(cobra)
})
resPlots <- plotsBif[seq(1,length(plotsBif),by=9)] # get relevant data frame


pAll <- ggplot(resPlots[[1]],
               aes(x = FDP, y = TPR, col = method)) +
  geom_path(size = 1, aes(linetype = method)) +
  scale_color_manual(values = cols, breaks = names(cols)) +
  scale_linetype_manual(values = linetypes, breaks = names(linetypes)) 

legend_all <- get_legend(pAll + labs(col = "", linetype = "") +
                           theme(legend.position = "bottom",
                            legend.text = element_text(size = 14)))
#legend.key.width = unit(1.5, "cm"),

pLeg <- plot_grid(pAll, legend_all, rel_heights=c(1,0.2), nrow=2, ncol=1)
pLeg

#### plot with trajectories
  FQnorm <- function(counts){
    rk <- apply(counts,2,rank,ties.method='min')
    counts.sort <- apply(counts,2,sort)
    refdist <- apply(counts.sort,1,median)
    norm <- apply(rk,2,function(r){ refdist[r] })
    rownames(norm) <- rownames(counts)
    return(norm)
  }
pal <- wes_palette("Zissou1", 12, type = "continuous")
dataAll <- readRDS("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/datasets_for_koen.rds")

for(datasetIter in c(1:10)){

  data <- dataAll[[datasetIter]]
  counts <- t(data$counts)

  # get milestones
  gid <- data$prior_information$groups_id
  gid <- gid[match(colnames(counts),gid$cell_id),]

  pal <- wes_palette("Zissou1", 12, type = "continuous")
  truePseudotime <- data$prior_information$timecourse_continuous
  g <- Hmisc::cut2(truePseudotime,g=12)

  # quantile normalization
  normCounts <- FQnorm(counts)

  ## dim red
  pca <- prcomp(log1p(t(normCounts)), scale. = FALSE)
  rd <- pca$x[,1:2]
  #plot(rd, pch=16, asp = 1, col=pal[g])

  library(princurve)
  pcc <- principal_curve(rd, smoother="periodic_lowess")
  #lines(x=pcc$s[order(pcc$lambda),1], y=pcc$s[order(pcc$lambda),2], col="red", lwd=2)


  ggTraj <- ggplot(as.data.frame(rd), aes(x = PC1, y = PC2)) +
    geom_point(col = pal[g], size=1) +
    theme_classic() +
    geom_path(x = pcc$s[order(pcc$lambda),1],
              y = pcc$s[order(pcc$lambda),2])
  assign(paste0("trajplot",datasetIter),ggTraj)
}

### fixed aspect ratio
p1 <- plot_grid(trajplot1 + coord_fixed(), trajplot2 + coord_fixed(), trajplot3 + coord_fixed(), trajplot4 + coord_fixed(), trajplot5 + coord_fixed(),
              bifplot1 + coord_fixed(), bifplot2 + coord_fixed(), bifplot3 + coord_fixed(), bifplot4 + coord_fixed(), bifplot5 + coord_fixed(),
        nrow=2, ncol=5)#, rel_heights=c(0.8,1,0.8,1))
pLeg1 <- plot_grid(p1, legend_all, rel_heights=c(1,0.15), nrow=2, ncol=1)
pLeg1
ggsave("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/individualPerformance_cyclic1To5_v2IncludingEdgeR.pdf", width = unit(15, "in"), height = unit(10, "in"), scale = .7)

## fixed aspect ratio
dev.new()
p2 <- plot_grid(trajplot6 + coord_fixed(), trajplot7 + coord_fixed(), trajplot8 + coord_fixed(), trajplot9 + coord_fixed(), trajplot10 + coord_fixed(),
                bifplot6 + coord_fixed(), bifplot7 + coord_fixed(), bifplot8 + coord_fixed(), bifplot9 + coord_fixed(), bifplot10 + coord_fixed(),
        nrow=2, ncol=5, rel_heights=c(0.8,1))
pLeg2 <- plot_grid(p2, legend_all, rel_heights=c(1,0.15), nrow=2, ncol=1)
pLeg2
ggsave("/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/individualPerformance_cyclic6To10_v2IncludingEdgeR.pdf", width = unit(15, "in"), height = unit(10, "in"), scale = .7)

# ### mean plot
resList <- c()
resList2 <- c()
alphaSeq <- c(seq(1e-16,1e-9,length=100),seq(5e-9,1e-3,length=250),seq(5e-2,1,length=500))
alphaSeq2 <- c(seq(-20,0,length=100),seq(0,3,length=250),seq(3,50,length=500))

for(ii in 1:length(cobraFiles)){
    cobra <- readRDS(cobraFiles[ii])
    pvals <- pval(cobra)
    colnames(pvals) <- gsub(colnames(pvals),pattern="tradeR",replacement="tradeSeq")
    truths <- as.logical(truth(cobra)[,1])
    # performance for all p-value based methods
    hlp <- apply(pvals,2,function(x){
      pOrder <- order(x,decreasing=FALSE)
      padj <- p.adjust(x,"fdr")
      df <- as.data.frame(t(sapply(alphaSeq, function(alpha){
          R <- which(padj <= alpha)
          fdr <- sum(!truths[R])/length(R) #false over rejected
          tpr <- sum(truths[R])/sum(truths) #TP over all true
          c(fdr=fdr, tpr=tpr, cutoff=alpha)
      })))
   
    })
    scores <- score(cobra) 
    # performance for all score-value based methods
    hlp2 <- apply(scores,2,function(x){
      pOrder <- order(scores[,1],decreasing=FALSE)
      df <- as.data.frame(t(sapply(alphaSeq2, function(alpha2){
        R <- which(x >= alpha2)
        fdr <- sum(!truths[R])/length(R) #false over rejected
        tpr <- sum(truths[R])/sum(truths) #TP over all true
        c(fdr=fdr, tpr=tpr, cutoff=alpha2)
      })))
     })
    
    # summarize
    dfIter <- do.call(rbind,hlp)
    dfIter2 <- do.call(rbind,hlp2)
    dfIter$method=rep(colnames(pvals),each=length(alphaSeq))
    dfIter2$method=rep(colnames(scores),each=length(alphaSeq))
    dfIter$dataset <- ii
    resList[[ii]] <- dfIter
    dfIter2$dataset <- ii
    resList2[[ii]] <- dfIter2
}

# #### across all datasets
library(tidyverse)
df <- as_tibble(do.call(rbind,resList))
df <- df %>% group_by(method,cutoff) %>%
        summarize(meanTPR=mean(tpr,na.rm=TRUE),
                meanFDR=mean(fdr,na.rm=TRUE)) %>% arrange(method,cutoff)

df2 <- as_tibble(do.call(rbind,resList2))
df2 <- df2 %>% group_by(method,cutoff) %>%
  summarize(meanTPR=mean(tpr,na.rm=TRUE),
            meanFDR=mean(fdr,na.rm=TRUE)) %>% arrange(method,cutoff)

pMeanAll <- ggplot(df, aes(x=meanFDR, y=meanTPR, col=method)) + geom_path(size = 1) +
scale_x_continuous(limits = c(0, 1), breaks = c(0.01, 0.05, 0.1, 0.5, 1),
                   minor_breaks = c(0:5) * .1) +
scale_y_continuous(limits = c(0, 1)) +
scale_color_manual(values = cols, breaks = names(cols)) +
scale_linetype_manual(values = linetypes, breaks = names(linetypes)) #+ xlab("FDR") + ylab("TPR")

pMeanLegAll <- plot_grid(pMeanAll, legend_all, rel_heights=c(1,0.15), nrow=2, ncol=1)
pMeanLegAll

pMeanAll2 <- ggplot(df2, aes(x=meanFDR, y=meanTPR, col=method)) + geom_path(size = 1) +
scale_x_continuous(limits = c(0, 1), breaks = c(0.01, 0.05, 0.1, 0.5, 1),
                   minor_breaks = c(0:5) * .1) +
scale_y_continuous(limits = c(0, 1)) +
scale_color_manual(values = cols, breaks = names(cols)) +
scale_linetype_manual(values = linetypes, breaks = names(linetypes))# + xlab("FDR") + ylab("TPR")

pMeanLegAll2 <- plot_grid(pMeanAll2, legend_all, rel_heights=c(1,0.15), nrow=2, ncol=1)
pMeanLegAll2

df_all <- rbind(df, df2)
pMeanAll3 <- ggplot(df_all, aes(x=meanFDR, y=meanTPR, col=method)) + geom_path(size = 1) +
  scale_x_continuous(limits = c(0, .6), breaks = c(0.01, 0.05, 0.1, 0.5, 1),
                     minor_breaks = c(0:5) * .1) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = cols, breaks = names(cols)) +
  scale_linetype_manual(values = linetypes, breaks = names(linetypes)) +xlab("") + ylab("")+theme(axis.text.x = element_text( 
                                   size=18, angle=-90), axis.text.y = element_text( size=18, angle=0))                   


#+ xlab("FDR") + ylab("TPR")
#pMeanLegAll3 <- plot_grid(pMeanAll3, legend_all, rel_heights=c(1,0.15), nrow=2, ncol=1) + guides(fill=guide_legend(nrow=2,byrow=TRUE)) 
pMeanLegAll3

saveRDS(pMeanAll3, file="/Users/nuhabintayyash/GPcounts/paper_notebooks/tradeSeq/pMeanCycle_v2_GPcounts.rds")
