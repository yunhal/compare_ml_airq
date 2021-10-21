#modified tayplor diagram----
# Display a Taylor diagram
# Taylor K.E. (2001)
# Summarizing multiple aspects of model performance in a single diagram
# Journal of Geophysical Research, 106: 7183-7192.

# version 1.0
# progr. Olivier.Eterradossi, 12/2007
# 2007-01-12 - modifications and Anglicizing - Jim Lemon
# version 2.0
# progr. initiale OLE, 8/01/2007
# rev. OLE 3/09/2008 : remove samples with NA's from mean, sd and cor calculations 
# 2008-09-04 - integration and more anglicizing - Jim Lemon
# 2008-12-09 - added correlation radii, sd arcs to the pos.cor=FALSE routine
# and stopped the pos.cor=FALSE routine from calculating arcs for zero radius
# Jim Lemon
# 2010-4-30 - added the gamma.col argument for pos.cor=TRUE plots - Jim Lemon
# 2010-6-24 - added mar argument to pos.cor=TRUE plots - Jim Lemon
# 2012-1-31 - added the cex.axis argument - Jim Lemon
# 2019-02-22 - added cex.axis to Olivier's text calls plus adj and srt

library(plotrix)
taylor.diagram<-function(ref,model,add=FALSE,col="red",pch=19,pos.cor = TRUE, 
                         xlab = "Standard deviation", ylab = "", main = "Taylor Diagram",
                         show.gamma = TRUE, ngamma = 3, gamma.col = 8, sd.arcs = 0, ref.sd = FALSE,
                         sd.method = "sample", grad.corr.lines = c(0.2, 0.4, 0.6, 0.8, 0.9), 
                         pcex = 1, cex.axis = 1, normalize = FALSE, mar = c(4, 3, 4, 3), xlim, ...) {
  
  grad.corr.full <- c(0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1)
  # convert any list elements or data frames to vectors
  R <- cor(ref, model, use = "pairwise")
  if (is.list(ref)) 
    ref <- unlist(ref)
  if (is.list(model)) 
    ref <- unlist(model)
  SD <- function(x, subn) {
    meanx <- mean(x, na.rm = TRUE)
    devx <- x - meanx
    ssd <- sqrt(sum(devx * devx, na.rm = TRUE)/(length(x[!is.na(x)]) - 
                                                  subn))
    return(ssd)
  }
  subn <- sd.method != "sample"
  sd.r <- SD(ref, subn)
  sd.f <- SD(model, subn)
  if (normalize) {
    sd.f <- sd.f/sd.r
    sd.r <- 1
  }
  maxsd <- xlim #1.5 * max(sd.f, sd.r)
  oldpar <- par("mar", "xpd", "xaxs", "yaxs")
  if (!add) {
    par(mar = mar)
    # display the diagram
    if (pos.cor) {
      if (nchar(ylab) == 0) 
        ylab = "Standard deviation"
      plot(0, xlim = c(0, maxsd*1.1), ylim = c(0, maxsd*1.1), xaxs = "i", 
           yaxs = "i", axes = FALSE, main = main, xlab = "", 
           ylab = ylab, type = "n", cex = cex.axis, ...)
      mtext(xlab,side=1,line=2.3)
      if (grad.corr.lines[1]) {
        for (gcl in grad.corr.lines) lines(c(0, maxsd * 
                                               gcl), c(0, maxsd * sqrt(1 - gcl^2)), lty = 3)
      }
      # add the axes
      segments(c(0, 0), c(0, 0), c(0, maxsd), c(maxsd, 
                                                0))
      axis.ticks <- pretty(c(0, maxsd))
      axis.ticks <- axis.ticks[axis.ticks <= maxsd]
      axis(1, at = axis.ticks,cex.axis=cex.axis)
      axis(2, at = axis.ticks,cex.axis=cex.axis)
      if (sd.arcs[1]) {
        if (length(sd.arcs) == 1) 
          sd.arcs <- axis.ticks
        for (sdarc in sd.arcs) {
          xcurve <- cos(seq(0, pi/2, by = 0.03)) * sdarc
          ycurve <- sin(seq(0, pi/2, by = 0.03)) * sdarc
          lines(xcurve, ycurve, col = "blue", lty = 3)
        }
      }
      if (show.gamma[1]) {
        # if the user has passed a set of gamma values, use that
        if (length(show.gamma) > 1) gamma <- show.gamma
        # otherwise make up a set
        else gamma <- pretty(c(0, maxsd), n = ngamma)[-1]
        if (gamma[length(gamma)] > maxsd) 
          gamma <- gamma[-length(gamma)]
        labelpos <- seq(45, 70, length.out = length(gamma))
        # do the gamma curves
        for (gindex in 1:length(gamma)) {
          xcurve <- cos(seq(0, pi, by = 0.03)) * gamma[gindex] + sd.r
          # find where to clip the curves
          endcurve <- which(xcurve < 0)
          endcurve <- ifelse(length(endcurve), min(endcurve) - 
                               1, 105)
          ycurve <- sin(seq(0, pi, by = 0.03)) * gamma[gindex]
          maxcurve <- xcurve * xcurve + ycurve * ycurve
          startcurve <- which(maxcurve > maxsd * maxsd)
          startcurve <- ifelse(length(startcurve), max(startcurve) + 
                                 1, 0)
          lines(xcurve[startcurve:endcurve], ycurve[startcurve:endcurve], 
                col = gamma.col)
          if (xcurve[labelpos[gindex]] > 0) 
            boxed.labels(xcurve[labelpos[gindex]], ycurve[labelpos[gindex]], 
                         gamma[gindex], border = FALSE)
        }
      }
      # the outer curve for correlation
      xcurve <- cos(seq(0, pi/2, by = 0.01)) * maxsd
      ycurve <- sin(seq(0, pi/2, by = 0.01)) * maxsd
      lines(xcurve, ycurve)
      bigtickangles <- acos(seq(0.1, 0.9, by = 0.1))
      medtickangles <- acos(seq(0.05, 0.95, by = 0.1))
      smltickangles <- acos(seq(0.91, 0.99, by = 0.01))
      segments(cos(bigtickangles) * maxsd, sin(bigtickangles) * 
                 maxsd, cos(bigtickangles) * 0.97 * maxsd, sin(bigtickangles) * 
                 0.97 * maxsd)
      par(xpd = TRUE)
      # the inner curve for reference SD
      if (ref.sd) {
        xcurve <- cos(seq(0, pi/2, by = 0.01)) * sd.r
        ycurve <- sin(seq(0, pi/2, by = 0.01)) * sd.r
        lines(xcurve, ycurve)
      }
      points(sd.r, 0, cex = pcex)
      text(cos(c(bigtickangles, acos(c(0.95, 0.99)))) * 
             1.05 * maxsd, sin(c(bigtickangles, acos(c(0.95, 
                                                       0.99)))) * 1.05 * maxsd, c(seq(0.1, 0.9, by = 0.1), 
                                                                                  0.95, 0.99),cex=cex.axis)
      text(maxsd * 0.8, maxsd * 0.8, "Correlation", srt = 315,cex=cex.axis)
      segments(cos(medtickangles) * maxsd, sin(medtickangles) * 
                 maxsd, cos(medtickangles) * 0.98 * maxsd, sin(medtickangles) * 
                 0.98 * maxsd)
      segments(cos(smltickangles) * maxsd, sin(smltickangles) * 
                 maxsd, cos(smltickangles) * 0.99 * maxsd, sin(smltickangles) * 
                 0.99 * maxsd)
    }
    else {
      x <- ref
      y <- model
      R <- cor(x, y, use = "pairwise.complete.obs")
      E <- mean(x, na.rm = TRUE) - mean(y, na.rm = TRUE)
      xprime <- x - mean(x, na.rm = TRUE)
      yprime <- y - mean(y, na.rm = TRUE)
      sumofsquares <- (xprime - yprime)^2
      Eprime <- sqrt(sum(sumofsquares)/length(complete.cases(x)))
      E2 <- E^2 + Eprime^2
      if (add == FALSE) {
        # pourtour du diagramme (display the diagram)
        maxray <- 1.5 * max(sd.f, sd.r)
        plot(c(-maxray, maxray), c(0, maxray), type = "n", 
             asp = 1, bty = "n", xaxt = "n", yaxt = "n", 
             xlim=c(-1.1*maxray,1.1*maxray),
             xlab = xlab, ylab = ylab, main = main, cex = cex.axis)
        discrete <- seq(180, 0, by = -1)
        listepoints <- NULL
        for (i in discrete) {
          listepoints <- cbind(listepoints, maxray * 
                                 cos(i * pi/180), maxray * sin(i * pi/180))
        }
        listepoints <- matrix(listepoints, 2, length(listepoints)/2)
        listepoints <- t(listepoints)
        lines(listepoints[, 1], listepoints[, 2])
        # axes x,y
        lines(c(-maxray, maxray), c(0, 0))
        lines(c(0, 0), c(0, maxray))
        # lignes radiales jusque R = +/- 0.8
        for (i in grad.corr.lines) {
          lines(c(0, maxray * i), c(0, maxray * sqrt(1 - 
                                                       i^2)), lty = 3)
          lines(c(0, -maxray * i), c(0, maxray * sqrt(1 - 
                                                        i^2)), lty = 3)
        }
        # texte radial
        for (i in grad.corr.full) {
          text(1.05 * maxray * i, 1.05 * maxray * sqrt(1 - 
                                                         i^2), i, cex = cex.axis,adj=cos(i)/2)
          text(-1.05 * maxray * i, 1.05 * maxray * sqrt(1 - 
                                                          i^2), -i, cex = cex.axis,adj=1-cos(i)/2)
        }
        # sd concentriques autour de la reference
        seq.sd <- seq.int(0, 2 * maxray, by = (maxray/10))[-1]
        for (i in seq.sd) {
          xcircle <- sd.r + (cos(discrete * pi/180) * 
                               i)
          ycircle <- sin(discrete * pi/180) * i
          for (j in 1:length(xcircle)) {
            if ((xcircle[j]^2 + ycircle[j]^2) < (maxray^2)) {
              points(xcircle[j], ycircle[j], col = "darkgreen", 
                     pch = ".")
              if (j == 10) 
                text(xcircle[j], ycircle[j], signif(i, 
                                                    2), cex = cex.axis, col = "darkgreen",srt=90)
            }
          }
        }
        # sd concentriques autour de l'origine
        seq.sd <- seq.int(0, maxray, length.out = 5)
        for (i in seq.sd) {
          xcircle <- cos(discrete * pi/180) * i
          ycircle <- sin(discrete * pi/180) * i
          if (i) 
            lines(xcircle, ycircle, lty = 3, col = "blue")
          text(min(xcircle), -0.06 * maxray, signif(i, 
                                                    2), cex = cex.axis, col = "blue")
          text(max(xcircle), -0.06 * maxray, signif(i, 
                                                    2), cex = cex.axis, col = "blue")
        }
        text(0, -0.14 * maxray, "Standard Deviation", 
             cex = cex.axis, col = "blue")
        text(0, -0.22 * maxray, "Centered RMS Difference", 
             cex = cex.axis, col = "darkgreen")
        points(sd.r, 0, pch = 22, bg = "darkgreen", cex = pcex)
        text(0, 1.2 * maxray, "Correlation Coefficient", 
             cex = cex.axis)
      }
      S <- (2 * (1 + R))/(sd.f + (1/sd.f))^2
      #   Taylor<-S
    }
  }
  # display the points
  points(sd.f * R, sd.f * sin(acos(R)), pch = pch, col = col,
         cex = pcex)
  invisible(oldpar)
}

#----
library(verification)
library(ggplot2)
library(reshape2)
library(lubridate)
library(MASS)
library(viridis)
require(scales)
library(hydroGOF)
library(ggmap)
register_google(key = "AIzaSyAVJVaZUYJ9NlacOPg5RhYK6nRy09Y4RC0")

models = c('AIRPACT','twoRF','denseXS','denseS','denseM','denseL','denseXL')
species = 'o3'
timelens = c('fullyear','warmmonths','coldmonths')
devices = c('p5split','p5split_less_features','p5split_less_features_minmaxlog','p5split_less_features_quantile','p65split','p8split')

plotdir_base='/bigdata/casus/atmos/play/DLair/code-compare-ml-dl/results_postproc/'
outdir_base='/home/lee45/play/DLair/code-compare-ml-dl/results_postproc/'

o3_ap <- read.csv(paste(plotdir_base,'daily_o3_AIRPACT.csv',sep = ''))

o3_ap$X <- as.Date(o3_ap$X)
colnames(o3_ap)[c(1:3)] <- c('date','AIRPACT','AQI_AIRPACT')

for (device in devices){
    for (timelen in timelens){
        rm(alldt)
        #read data
        for (m in models[2:length(models)]){
            tmp <- read.csv(paste(plotdir_base, 'daily_',species,'_',m,'_',timelen,'_',device,'.csv',sep=''))
            colnames(tmp) <- c('date','obs',m,'AQI_obs',paste('AQI_',m,sep=''),'site')
            tmp$date <- as.Date(tmp$date)
            if (exists("alldt")) alldt <- merge(alldt,tmp, by=c('date','obs','AQI_obs','site'))
            else alldt <- tmp
        }
        alldt <- merge(alldt,o3_ap,, by=c('date','site'))
        colnames(alldt)[c(5,6)] <- c('twoRF','AQI_twoRF') #avoid using leading number
        #'rename 2rf twoRF *' can change filenames in linux
        
        #plot nmb
        stat <- data.frame(matrix(NA, ncol = 8))
        colnames(stat) <- c('Site',models)
        for (s in 1:length(unique(alldt$site))){
            tmp <- alldt[alldt$site == unique(alldt$site)[s],]
            stat[s,'Site'] <- unique(alldt$site)[s]
            for (m in models){
                stat[s,m] <- sum(tmp[,m] - tmp$obs,na.rm = T)/sum(tmp$obs,na.rm = T)*100
            }
        }


        ggplot()+
          geom_boxplot(data=melt(stat,id='Site'), aes_string(x='variable', y='value', group = 'variable'),outlier.shape=NA)+
          geom_hline(yintercept=0, color = "red")+
          ylab('NMB (%)') +
          xlab('')+
          scale_y_continuous(limits = c(-100, 200))+
          theme(legend.position = 'right',legend.title=element_text(size=12),
                axis.text=element_text(size=15,angle=90),
                axis.title=element_text(size=20,face="bold"), legend.text=element_text(size=12),
                plot.title = element_text(hjust = 0.5,size=20,face="bold"),
                strip.text.x = element_text(size=15),
                strip.background = element_rect(colour=NA, fill=NA))+
          theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                panel.background = element_blank(), axis.line = element_line(colour = "black"),
                panel.border = element_rect(colour = "black", fill=NA, size=1))
        ggsave(paste(outdir_base,'NMB_',species,'_',timelen,'_',device,'.png',sep=''), units="in", width=6, height=4,dpi=300)
        
        #plot nme
        stat <- data.frame(matrix(NA, ncol = 8))
        colnames(stat) <- c('Site',models)
        for (s in 1:length(unique(alldt$site))){
            tmp <- alldt[alldt$site == unique(alldt$site)[s],]
            stat[s,'Site'] <- unique(alldt$site)[s]
            for (m in models){
                stat[s,m] <- sum(abs(tmp[,m] - tmp$obs),na.rm = T)/sum(tmp$obs,na.rm = T)*100
            }
        }


        ggplot()+
          geom_boxplot(data=melt(stat,id='Site'), aes_string(x='variable', y='value', group = 'variable'),outlier.shape=NA)+
          ylab('NME (%)') +
          xlab('')+
          #scale_y_continuous(limits = c(-100, 200))+
          theme(legend.position = 'right',legend.title=element_text(size=12),
                axis.text=element_text(size=15,angle=90),
                axis.title=element_text(size=20,face="bold"), legend.text=element_text(size=12),
                plot.title = element_text(hjust = 0.5,size=20,face="bold"),
                strip.text.x = element_text(size=15),
                strip.background = element_rect(colour=NA, fill=NA))+
          theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                panel.background = element_blank(), axis.line = element_line(colour = "black"),
                panel.border = element_rect(colour = "black", fill=NA, size=1))
        ggsave(paste(outdir_base,'NME_',species,'_',timelen,'_',device,'.png',sep=''), units="in", width=6, height=4,dpi=300)
        
        #plot scatter
        get_density <- function(x, y, ...) {
          dens <- MASS::kde2d(x, y, ...)
          ix <- findInterval(x, dens$x)
          iy <- findInterval(y, dens$y)
          ii <- cbind(ix, iy)
          return(dens$z[ii])
        }

        for (m in models){
            lmap = lm(formula = alldt[, m] ~ alldt$obs) #Create a linear regression with two variables
            r2 = round(summary(lmap)$adj.r.squared, 2)
            nmb = round(sum(alldt[, m] - alldt$obs,na.rm = T)/sum(alldt$obs,na.rm = T)*100,0)
            nme = round(sum(abs(alldt[, m] - alldt$obs),na.rm = T)/sum(alldt$obs,na.rm = T)*100,0)

            alldt$density <- get_density(alldt$obs, alldt[, m], n = 2000)
            ggplot(alldt)+
              geom_point(aes_string(x="obs",y=m,color="density"))+
              geom_abline(intercept = 0, slope = 1, size=1, col='red')+
              #geom_abline(intercept = 35.5, slope = 0, size=1, col='blue')+
              geom_hline(yintercept = 35.5, color = "blue", size=1)+
              geom_vline(xintercept = 35.5, color = "blue", size=1)+
              #ggtitle("(a) AP")+
              xlab('Obs. MDA8 O3 (ppb)')+
              ylab(paste(m,' MDA8 O3 (ppb)'))+
              #scale_x_continuous(trans = 'log10',breaks=c(1,10,100), limits=c(0.1,650)) +
              #scale_y_continuous(trans = 'log10',breaks=c(1,10,100), limits=c(0.1,650)) +
              theme(legend.position = 'none',legend.title=element_text(size=20),
                    axis.text=element_text(size=20),
                    axis.title=element_text(size=25,face="bold"), legend.text=element_text(size=20),
                    plot.title = element_text(hjust = 0.5,size=25,face="bold"),
                    strip.text.x = element_text(size=20),
                    strip.background = element_rect(colour=NA, fill=NA))+
              theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                    panel.background = element_blank(), axis.line = element_line(colour = "black"),
                    panel.border = element_rect(colour = "black", fill=NA, size=1))+
              annotate("text", x = 5, y=65, size = 6, hjust=0, col='red',
                       label = paste("R2 : ",r2,"\nNMB: ",nmb, "%\nNME: ",nme, "%", sep=''))

            ggsave(paste(outdir_base,'Scatter_',species,'_',m,'_',timelen,'_',device,'.png',sep=''), units="in", width=5, height=5,dpi=300)
        }

        #plot taylor diagram
        for (m in models){
          png(paste(outdir_base,'Taylor_',species,'_',m,'_',timelen,'_',device,'.png',sep=''), units="in", width=5, height=5, res=300)
          taylor.diagram(alldt[,'obs'],alldt[,'obs'],add=FALSE,col="black",pch=1,pos.cor=TRUE,
                         xlab="Standard deviation",ylab="",main="",
                         show.gamma=TRUE,ngamma=3,gamma.col=8,sd.arcs=0,
                         ref.sd=FALSE,sd.method="sample",grad.corr.lines=c(0.2,0.4,0.6,0.8,0.9),
                         pcex=1,cex.axis=1,normalize=TRUE,mar=c(4,3,4,3),xlim=2)
          legend(3.5,4.1,legend=c("twoRF",m),pch=1,col=c("red","blue"))
          for (s in unique(alldt$site)){
            taylor.diagram(alldt[alldt$site==s,'obs'],alldt[alldt$site==s,'twoRF'],add=TRUE,col="green",pch=1,pos.cor=TRUE,
                           xlab="Standard deviation",ylab="",main="",
                           show.gamma=TRUE,ngamma=3,gamma.col=8,sd.arcs=0,
                           ref.sd=FALSE,sd.method="sample",grad.corr.lines=c(0.2,0.4,0.6,0.8,0.9),
                           pcex=1,cex.axis=1,normalize=TRUE,mar=c(4,3,4,3),xlim=2)
            taylor.diagram(alldt[alldt$site==s,'obs'],alldt[alldt$site==s,m],add=TRUE,col="blue",pch=1,pos.cor=TRUE,
                           xlab="Standard deviation",ylab="",main="",
                           show.gamma=TRUE,ngamma=3,gamma.col=8,sd.arcs=0,
                           ref.sd=FALSE,sd.method="sample",grad.corr.lines=c(0.2,0.4,0.6,0.8,0.9),
                           pcex=1,cex.axis=1,normalize=TRUE,mar=c(4,3,4,3),xlim=2)
            taylor.diagram(alldt[alldt$site==s,'obs'],alldt[alldt$site==s,'AIRPACT'],add=TRUE,col="red",pch=1,pos.cor=TRUE,
                           xlab="Standard deviation",ylab="",main="",
                           show.gamma=TRUE,ngamma=3,gamma.col=8,sd.arcs=0,
                           ref.sd=FALSE,sd.method="sample",grad.corr.lines=c(0.2,0.4,0.6,0.8,0.9),
                           pcex=1,cex.axis=1,normalize=TRUE,mar=c(4,3,4,3),xlim=2)
          }
          dev.off()
        }
        
        #plot ioa map
        site_info <- read.csv('/bigdata/casus/atmos/ML_model/Evaluation/Unique_monitoring_sites_and_locations_for_all_pollutants.csv', stringsAsFactors=FALSE)
        site_info$AQS_ID <- formatC(as.numeric(site_info$AQS_ID), width = 9, format = "d", flag = "0")
        alldt <- merge(alldt,site_info,by.x='site',by.y='AQS_ID')
        sites <- unique(alldt$site)

        ioa <- data.frame(matrix(NA,nrow=length(unique(alldt$site)),ncol=9))
        colnames(ioa) <- c('AQS_ID','OBS_mean',models)
        for (s in 1:length(sites)){
          sub_df <- alldt[alldt$site==sites[s],]
          sub_df <- sub_df[complete.cases(sub_df),]
          ioa[s,'AQS_ID'] <- sites[s]
          ioa[s,'OBS_mean'] <- mean(sub_df$obs)
          for (m in models){
            ioa[s,m] <- d(sub_df[,m],sub_df$obs,na.rm = T)
          }
        }

        ioa <- merge(site_info, ioa, by='AQS_ID')

        
        test <- get_map(location = c(-117.8,44.5), maptype = 'terrain',
                        zoom = 6)
        jet.colors0 <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "white", "yellow", "#FF7F00", "red", "#7F0000"))

        for (m in models){
          a <- ggmap(test) +
            #coord_map()+
            # for values at site locations
            geom_point(data = ioa,aes_string(x='Longitude', y ='Latitude', size = 'OBS_mean', fill = m #, 
                                             #shape=location_setting
            ), alpha=0.7, shape=21)+
            #for pm2.5
            scale_size_continuous("MDA8 O3 Obs. Mean\n  (ppb)",
                                  #limits=c(0,15),breaks=c(1,2,5,10),range = c(1,15)
                                  range = c(4,8)
            )+
            #NMBs
            # using jet colors
            scale_fill_gradientn("IOA",colours = jet.colors0(7), #c("#00007F", "#002AFF", "#00D4FF", "#FFFFFF"), #
                                 breaks=seq(0, 1, by=0.2),limits=c(0,1), oob=squish) +
            #breaks=ceiling(seq(-100, -30, by=20)),limits=c(-100,-30), oob=squish) +
            #scale_shape_manual("Location Setting",values = c(21, 22, 24))+
            # ordering what legent you want to show first and second
            guides( size = guide_legend(order = 1, size =7),
                    fill = guide_colourbar(order = 2, barwidth = 20, #barwidth = 2, barheight = 23, #
                                           raster = FALSE, ticks = TRUE,
                                           draw.ulim = TRUE, draw.llim = TRUE),
                    shape = guide_legend(order = 3,override.aes = list(size=4))) +
            theme_bw() +
            #without labels in axis and texts in tickmarks 
            theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
            theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank()) +
            theme(panel.background = element_rect(colour = "black", size = 1.5)) +
            # outline around legent boxes
            theme(legend.position = "bottom",legend.spacing  = unit(0.5, "cm"),
                  legend.box = "vertical") +
            theme(plot.title = element_text(size = 15, lineheight=.8, face="bold"),
                  legend.title=element_text(size = 14,face="bold"),
                  legend.text=element_text(face="bold",size = 14)) 

          png(paste(outdir_base,'IOA_map_',species,'_',m,'_',timelen,'_',device,'.png',sep=''), units="in", width=7, height=5, res=300)
          print(a)
          dev.off()
        }
        
        #plot nmb map
        site_info <- read.csv('/bigdata/casus/atmos/ML_model/Evaluation/Unique_monitoring_sites_and_locations_for_all_pollutants.csv', stringsAsFactors=FALSE)
        site_info$AQS_ID <- formatC(as.numeric(site_info$AQS_ID), width = 9, format = "d", flag = "0")
        alldt <- merge(alldt,site_info,by.x='site',by.y='AQS_ID')
        sites <- unique(alldt$site)

        nmb <- data.frame(matrix(NA,nrow=length(unique(alldt$site)),ncol=9))
        colnames(nmb) <- c('AQS_ID','OBS_mean',models)
        for (s in 1:length(sites)){
          sub_df <- alldt[alldt$site==sites[s],]
          sub_df <- sub_df[complete.cases(sub_df),]
          nmb[s,'AQS_ID'] <- sites[s]
          nmb[s,'OBS_mean'] <- mean(sub_df$obs)
          for (m in models){
            nmb[s,m] <- sum(sub_df[,m]-sub_df$obs,na.rm = T)/sum(sub_df$obs,na.rm = T)*100
          }
        }

        nmb <- merge(site_info, nmb, by='AQS_ID')

        
        test <- get_map(location = c(-117.8,44.5), maptype = 'terrain',
                        zoom = 6)
        jet.colors0 <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "white", "yellow", "#FF7F00", "red", "#7F0000"))

        for (m in models){
          a <- ggmap(test) +
            #coord_map()+
            # for values at site locations
            geom_point(data = nmb,aes_string(x='Longitude', y ='Latitude', size = 'OBS_mean', fill = m #, 
                                             #shape=location_setting
            ), alpha=0.7, shape=21)+
            #for pm2.5
            scale_size_continuous("MDA8 O3 Obs. Mean\n  (ppb)",
                                  #limits=c(0,15),breaks=c(1,2,5,10),range = c(1,15)
                                  range = c(4,8)
            )+
            #NMBs
            # using jet colors
            scale_fill_gradientn("NMB (%)",colours = jet.colors0(7), #c("#00007F", "#002AFF", "#00D4FF", "#FFFFFF"), #
                                 breaks=seq(-30, 30, by=10),limits=c(-30,30), oob=squish) +
            #breaks=ceiling(seq(-100, -30, by=20)),limits=c(-100,-30), oob=squish) +
            #scale_shape_manual("Location Setting",values = c(21, 22, 24))+
            # ordering what legent you want to show first and second
            guides( size = guide_legend(order = 1, size =7),
                    fill = guide_colourbar(order = 2, barwidth = 20, #barwidth = 2, barheight = 23, #
                                           raster = FALSE, ticks = TRUE,
                                           draw.ulim = TRUE, draw.llim = TRUE),
                    shape = guide_legend(order = 3,override.aes = list(size=4))) +
            theme_bw() +
            #without labels in axis and texts in tickmarks 
            theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
            theme(axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank()) +
            theme(panel.background = element_rect(colour = "black", size = 1.5)) +
            # outline around legent boxes
            theme(legend.position = "bottom",legend.spacing  = unit(0.5, "cm"),
                  legend.box = "vertical") +
            theme(plot.title = element_text(size = 15, lineheight=.8, face="bold"),
                  legend.title=element_text(size = 14,face="bold"),
                  legend.text=element_text(face="bold",size = 14)) 

          png(paste(outdir_base,'NMB_map_',species,'_',m,'_',timelen,'_',device,'.png',sep=''), units="in", width=7, height=5, res=300)
          print(a)
          dev.off()
        }
        
        #obs table
        obs <- alldt
        obs$Year <- year(obs$date)
        summary_obs <- data.frame(matrix(data=NA,nrow=3,ncol=9))
        colnames(summary_obs) <- c('Year','site_num','mean','aqi1','aqi2','aqi3','aqi4','aqi5','aqi6')
        summary_obs[,c(1,3)] <- round(aggregate(obs$obs, by=list(Category=obs$Year), FUN=mean),1)

        for (i in 2018:2020){
          tmp <- obs[year(obs$date)==i,]
          summary_obs[i-2017,'site_num'] <- length(unique(tmp$site))
          for (q in 1:6){
            summary_obs[i-2017,q+3] <- nrow(tmp[tmp$AQI_obs==q,])
          }
        }

        write.csv(summary_obs,paste(outdir_base,'O3_obs_',timelen,'_',device,'.csv',sep=''),row.names = FALSE)

        #stat table
        stat <- data.frame(matrix(NA,nrow=12,ncol=7))
        colnames(stat) <- models
        rownames(stat) <- c('R2','NMB','NME','IOA','HSS','KSS','CSI1','CSI2','CSI3','CSI4','CSI5','CSI6')
        for (m in 1:7){
          lmap <- lm(alldt[,colnames(stat)[m]]~alldt$obs) #Create a linear regression with two variables
          stat[1,m] <- signif(summary(lmap)$adj.r.squared, 2)
          stat[2,m] <- signif(sum(alldt[,colnames(stat)[m]] - alldt$obs,na.rm = T)/sum(alldt$obs,na.rm = T)*100,2)
          stat[3,m] <- signif(sum(abs(alldt[,colnames(stat)[m]] - alldt$obs),na.rm = T)/sum(alldt$obs,na.rm = T)*100,2)
          stat[4,m] <- signif(d(alldt[,colnames(stat)[m]],alldt$obs,na.rm = T),2)

          stat_table <- data.frame(matrix(NA,nrow=6,ncol=6))
          colnames(stat_table) <- c('obs_1','obs_2','obs_3','obs_4','obs_5','obs_6')
          aqiname <- paste('AQI',colnames(stat)[m],sep='_')
          for (i in 1:6){
            stat_table[i,'obs_1'] <- nrow(alldt[alldt$AQI_obs==1&alldt[,aqiname]==i,])
            stat_table[i,'obs_2'] <- nrow(alldt[alldt$AQI_obs==2&alldt[,aqiname]==i,])
            stat_table[i,'obs_3'] <- nrow(alldt[alldt$AQI_obs==3&alldt[,aqiname]==i,])
            stat_table[i,'obs_4'] <- nrow(alldt[alldt$AQI_obs==4&alldt[,aqiname]==i,])
            stat_table[i,'obs_5'] <- nrow(alldt[alldt$AQI_obs==5&alldt[,aqiname]==i,])
            stat_table[i,'obs_6'] <- nrow(alldt[alldt$AQI_obs==6&alldt[,aqiname]==i,])
          }
          stat[5,m] <- signif(multi.cont(as.matrix(stat_table))$hss, 2)
          stat[6,m] <- signif(multi.cont(as.matrix(stat_table))$pss, 2)
          stat[7,m] <- signif(multi.cont(as.matrix(stat_table))$ts[1], 2)
          stat[8,m] <- signif(multi.cont(as.matrix(stat_table))$ts[2], 2)
          stat[9,m] <- signif(multi.cont(as.matrix(stat_table))$ts[3], 2)
          stat[10,m] <- signif(multi.cont(as.matrix(stat_table))$ts[4], 2)
          stat[11,m] <- signif(multi.cont(as.matrix(stat_table))$ts[5], 2)
          stat[12,m] <- signif(multi.cont(as.matrix(stat_table))$ts[6], 2)
        }

        write.csv(stat,paste(outdir_base,'O3_stats_',timelen,'_',device,'.csv',sep=''),
                  row.names = TRUE)
    }
}
