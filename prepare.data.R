############################################################
### Prepare data set for DNN Training with "BBPredNet"
### (Appendix: Predicting bark beetle outbreak dynamics with a Deep Neural Network)
### Werner Rammer, Rupert Seidl
### 
### Data from: Small beetle, large-scale drivers: how regional and landscape factors affect outbreaks of the European spruce bark beetle
### Seidl R, Müller J, Hothorn T, Bässler C, Heurich M, Kautz M
### Date Published: September 24, 2015
### DOI: http://dx.doi.org/10.5061/dryad.c5g9s
############################################################
library(dplyr)
library(raster)

# downloaded from http://datadryad.org/resource/doi:10.5061/dryad.c5g9s
# see also the README.txt file for details!
data <- read.csv("data/data.txt", sep="")


## number of damaged cells per year
outbr.yr <-  data %>% filter(bb.kill>0) %>% group_by(year) %>% summarise(n=n())


############################################################
## Convert data to grids for all years of the outbreak #####
############################################################
all.grids <- stack()
for (yr in outbr.yr$year) {
  tst <- data %>% filter(year==yr)
  # create a raster for bark beetle damage of the year
  cdf <- data.frame(x=tst$x.coord, y=tst$y.coord, z=tst$bb.kill)
  r <- rasterFromXYZ(cdf)

  if (nlayers(all.grids)>0) {
     ### later years have a smaller extent -> increase size to match other grids
     r <- extend(r, extent(all.grids[[1]]))
  }
    
  # save the created rasters to the RasterStack
  all.grids <- addLayer(all.grids, c(yr=r))

}

names(all.grids) <- outbr.yr$year


# Add the information of damages of previous years:
# code: 1: damage in the curent year, 2: damage in previous year, 3: damage two years ago
all.grids.prep <- all.grids
all.grids.prep[[2]] <-  max(all.grids[[2]]*1,all.grids[[2-1]]*2, na.rm=T) 
for (i in 3:nlayers(all.grids.prep)) {
  all.grids.prep[[i]] <-  max(all.grids[[i]]*1,all.grids[[i-1]]*2,all.grids[[i-2]]*3, na.rm=T) 
}
names(all.grids.prep) <- names(all.grids)

#plot(all.grids.prep) 


###################################
#### Prepare climate proxies ######
###################################

### long-term temperature grid
tst <- data %>% filter(year==1990) # note: the temperature is a long-term mean, and equal for all years
cdf <- data.frame(x=tst$x.coord, y=tst$y.coord, z=tst$S3.temp.yr)
temp.grid <- rasterFromXYZ(cdf)
plot(temp.grid)


### there are only minimal spatial differenecs in precip pattern -> use a single value for the whole area for every year
prec.anomaly <- (data %>% group_by(year) %>% summarise(anomaly=mean(S1.JJA.prec)))$anomaly
names(prec.anomaly) <- outbr.yr$year


############################
#### Extract examples ######
############################
# a single example for the training is a potential host cell, and the surrounding pixels.
# The neighborhood is an "image" with 19x19 pixels (a square with 570m)
# In addition, each sample includes the long-term mean annual temperature of the cell (S3.temp.yr), the
# summer precipitation anomaly (of the year) (S1.JJA.prec), and the regional outbreak stage (S1.outbreak), given in three classes.

## prepare offsets: 19x19 px, 1 center pixel, 9x30m = 270m in each direction
# the offsets are relative to a center pixel 
#r<-all.grids.prep[[1]]
#ncol(r)
dimpx <- 9
indices <- rep(NA, (dimpx*2+1)^2)
i<-1
for (x in -dimpx:dimpx)
  for (y in -dimpx:dimpx) {
    indices[i] <- x*ncol(r)+y
    i<-i+1
  }
    


all.points <- data.frame()
for (yr in outbr.yr$year) {
  print(yr)
  r <- all.grids.prep[[paste("X",yr,sep="")]]
  pts <- which(r[]==1) # damage in current year
  no.pts <- which(r[]==0) ## no damage in current year, but potential host
  
  # extract information for all damaged pixels...
  tst <- sapply(pts, function(p) r[p + indices]) # extract 361 values from the raster, relative to each point
  pts.df <- as.data.frame(t(tst))
  
  ## ... and for all undamaged pixels
  tst <- sapply(no.pts, function(p) r[p + indices]) 
  pts.df <- rbind(pts.df, as.data.frame(t(tst)))
  
  # add proxies
  pts.df$temp <- temp.grid[][c(pts, no.pts)]
  pts.df$damage <- c(rep(1, length(pts)), rep(0, length(no.pts)))
  pts.df$year <- yr
  pts.df$precanomaly <- prec.anomaly[yr-1989]
  pts.df$pts <- c(pts, no.pts) ## index
  all.points <- rbind(all.points, pts.df)
}

all.points$year <- all.points$year - 1989 ### make year to start with 1990=1, 1991=2, ...
all.points$V181 <- 0 ### drop the information of the center pixel (target for learning) (19x19 px)

# the examples that are used for training should only contain
# damages from previous years (as we want to predict damage solely from the state of the previous year(s))
# Coding: 0: non-host pixel (NA), 1: host pixels, 2: damage in the previous year, 3: damage 2 years ago
all.points[,1:361][all.points[,1:361] == 0] <- 1 
all.points[is.na(all.points)] <- 0

## fit into 8-bit integer (just to speed things up)
# the learning algorithm does not care about units, so we simply
# squeeze the data into a range (0..127)
all.points$temp <- as.integer(all.points$temp * 10)
all.points$precanomaly <- as.integer(all.points$precanomaly / 5) 

## add the outbreak_level as a numeric value
outbreak_level <- data[data$ID==1,c("year", "S1.outbreak")]
outbreak_level$year <- outbreak_level$year - 1989
outbreak_code <- as.integer(outbreak_level$S1.outbreak)
all.points$code <- NA
all.points$code <- outbreak_code[all.points$year]


############################
###### Save data sets ######
############################


## random shuffle, then split into 2 data sets
all.points.shf <- all.points[sample(nrow(all.points)), ]

#### Experiment 1 ########
# set aside individual years
# set aside 1993 (background), 1997 (gradation), 2005 (culmination)
write.table(all.points.shf[(all.points.shf$year+1989) %in% c(1993,1997,2005),], "bbyearseval.txt", row.names = F, col.names=F, sep=" ")
write.table(all.points.shf[!((all.points.shf$year+1989) %in% c(1993,1997,2005)),], "bbyearstrain.txt", row.names = F, col.names=F, sep=" ")


#### Experiment 2 ########
# use a random selection of 20% for testing (N=373817), 80% for training.
write.table(all.points.shf[1:nrow(all.points)*0.8,], "bbtrainall.txt", row.names = F, col.names=F, sep=" ")
write.table(all.points.shf[(nrow(all.points)*0.8+1):nrow(all.points),], "bbevalall.txt", row.names = F, col.names=F, sep=" ")



