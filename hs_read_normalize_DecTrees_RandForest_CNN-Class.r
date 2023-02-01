################################################################################
# Normalization of HyperSpec Data
# Author: Jared Streich
# Date Created Sept 8th 2022
# Version 0.1.0
# email: ju0@ornl.gov, not at ornl: streich.jared@gmail.com
################################################################################


################################################################################
############################## Libraries #######################################
################################################################################

library(imager)
library(raster)
library(dplyr)
library(stats)
library(randomForest)
library(keras)
library(tensorflow)
library(stringr)
library(readr)
library(purrr)
library(rpart)
library(rpart.plot)
library(e1071)
library(tidyverse)
library(caret)


# install.packages("e1071")
# install.packages("keras")
# install.packages("tensorflow")
# install.packages("stats")
# install.packages("dplyr")
# install.packages("randomForest")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("mlbench")
# install.packages("tidyverse")
# install.packages("caret")

################################################################################
########################### Load Color Palette #################################
################################################################################

#### Color gradient
cols <- colorRampPalette(c("grey30", "royalblue3", "palegreen3", "goldenrod3", "khaki2"))(100)

################################################################################
############################## Load Data #######################################
################################################################################

##### Set Directory
setwd("~/Desktop/hs/all-csv/")

##### Read in treatment/control data
id.trt <- read.csv("labels.txt", header = T)
id.trt[1:5,]
dim(id.trt)

##### Read in raw csv files
setwd("raw_extracted_csv/")
fls <- list.files(pattern = "*.csv")
length(fls)

i <- 1
### for loop to read data
for(i in 1:length(fls)){
  fls.i <- read.csv(fls[i], header = T)
  head(fls.i)
  dim.fls <- c(fls[i], nrow(fls.i))
  if(i == 1){
    dim.fls.p <- dim.fls
  }
  else{
    dim.fls.p <- rbind(dim.fls.p, dim.fls)
    colnames(dim.fls.p) <- c("file_name", "n_pixels")
  }
}

##### Sort by smallest leaf
hist(as.numeric(dim.fls.p[,2]), breaks = 50)
mean(as.numeric(dim.fls.p[,2]))
median(as.numeric(dim.fls.p[,2]))

hist(-log10(as.numeric(dim.fls.p[,2])/143000), breaks = 50)

dim.fls.sort <- dim.fls.p[order(as.numeric(dim.fls.p[,2]), decreasing = F), ]
dim.fls.sort[1,]
dim.fls.sort[nrow(dim.fls.sort),]

mx.samp.80 <- round(as.numeric(dim.fls.sort[1,2])*0.8)


##### Normalize 0-1 column wise
i <- 1
j <- 4
for(i in 1:length(fls)){
  fls.i <- read.csv(dim.fls.sort[i,1], header = T)
  fls.i[1:5,1:10]
  for(j in 4:ncol(fls.i)){
    fls.col <- as.numeric(fls.i[,j])
    fls.col.min <- min(fls.col)
    fls.col <- fls.col - fls.col.min
    fls.col.max <- max(fls.col)
    max.adj <- 1/fls.col.max
    fls.col <- fls.col*max.adj
    range(fls.col)
    fls.i[,j] <- as.numeric(fls.col)
  }
  fls.nm <- rep(dim.fls.sort[i,1], times = nrow(fls.i))
  fls.i <- cbind(fls.nm, fls.i)
  write.csv(fls.i, file = paste("nrm0-1_", dim.fls.sort[i,1], sep = ""))
}


##### Normalize 0-1 column wise with subsampling without replacement
i <- 1
j <- 4
for(i in 1:length(fls)){
  fls.i <- read.csv(dim.fls.sort[i,1], header = T)
  fls.i[1:5,1:10]
  for(j in 4:ncol(fls.i)){
    fls.col <- as.numeric(fls.i[,j])
    fls.col.min <- min(fls.col)
    fls.col <- fls.col - fls.col.min
    fls.col.max <- max(fls.col)
    max.adj <- 1/fls.col.max
    fls.col <- fls.col*max.adj
    range(fls.col)
    fls.i[,j] <- as.numeric(fls.col)
  }
  fls.i <- fls.i[sample(nrow(fls.i), mx.samp.80), ]
  fls.nm <- rep(dim.fls.sort[i,1], times = nrow(fls.i))
  fls.i <- cbind(fls.nm, fls.i)
  write.csv(fls.i, file = paste("nrm0-1_maxSamp-", mx.samp.80, "pxls_SampID-", dim.fls.sort[i,1], sep = ""))
}



##### Normalize 0-1 column wise with subsampling without replacement
i <- 1
j <- 4
for(i in 1:length(fls)){
  fls.i <- read.csv(dim.fls.sort[i,1], header = T)
  fls.i[1:5,1:10]
  for(j in 4:ncol(fls.i)){
    fls.col <- as.numeric(fls.i[,j])
    fls.col.min <- min(fls.col)
    fls.col <- fls.col - fls.col.min
    fls.col.max <- max(fls.col)
    max.adj <- 1/fls.col.max
    fls.col <- fls.col*max.adj
    range(fls.col)
    fls.i[,j] <- as.numeric(fls.col)
  }
  fls.i <- fls.i[sample(nrow(fls.i), size = 50000, replace = T), ]
  fls.nm <- rep(dim.fls.sort[i,1], times = nrow(fls.i))
  fls.i <- cbind(fls.nm, fls.i)
  write.csv(fls.i, file = paste("nrm0-1_maxSampWithReplace-", "50k", "pxls_SampID-", dim.fls.sort[i,1], sep = ""))
}


### Dosage normalization with extra sample files in place of
for(i in 1:length(fls)){
  fls.i <- read.csv(dim.fls.sort[i,1], header = T)
  dim(fls.i)
  fls.i[1:5,1:10]
  for(j in 4:ncol(fls.i)){
    fls.col <- as.numeric(fls.i[,j])
    fls.col.min <- min(fls.col)
    fls.col <- fls.col - fls.col.min
    fls.col.max <- max(fls.col)
    max.adj <- 1/fls.col.max
    fls.col <- fls.col*max.adj
    range(fls.col)
    fls.i[,j] <- as.numeric(fls.col)
  }
  nfls <- round(50000/nrow(fls.i))
  ran.nums <- sample(c(1:9999), size = 9999, replace = F)
  if(nfls > 1){
    set.seed(ran.nums[i])
    for(l in 1:nfls){
      fls.i <- fls.i[sample(nrow(fls.i), size = mx.samp.80, replace = F), ]
      fls.nm <- rep(dim.fls.sort[i,1], times = nrow(fls.i))
      fls.i <- cbind(fls.nm, fls.i)
      write.csv(fls.i, file = paste("nrm0-1_maxSampWithOutReplace-", l,"_", mx.samp.80, "pxls_SampID-", dim.fls.sort[i,1], sep = ""))
    }
  }
}


image(t(as.matrix(fls.i[,4:ncol(fls.i)])), col = cols)


##### Read in Example Data
hs.csv <- read.csv("232_p34_l2.csv", header = T)
hs.csv <- fls.i

##### Read in Raw file
hs.rw <- read.delim("232.raw", header = F)
hs.dt <- raster("REFLECTANCE_232.dat", band = 1)

dim(hs.rw)

################################################################################
############################# Start Script #####################################
################################################################################

########## View Data and get parameters and sizes ##########
##### Glance at data
head(hs)

##### Dim of data
dim(hs)

##### View Simple heat map of data
image(as.matrix(hs.csv))
plot(as.matrix(as.numeric(hs.csv[,3])), type = "l")

##### Check image dimentions
dim(hs)



##### Create Matrix from segmented image
hs.mat <- matrix(data = 0, nrow = 512, ncol = 512)
dim(hs.mat)

i <- 1
for(i in 1:nrow(hs.csv)){
  spct.val <- hs.csv[i,4]
  hs.mat[hs.csv[i,2], hs.csv[i,3]] <- hs.csv[i,4]
}

image(hs.mat, col = cols)

max(hs.mat)
min(hs.mat)


################################################################################
#################### Loop through csvs for min-mean-max ########################
################################################################################

##### Change directory to list of files
setwd("~/Desktop/hs/all-csv/raw_extracted_csv/")

##### Get list of file names
file.list <- list.files(pattern = ".csv")
file.list

i <- 1
j <- 4
for(i in 1:length(file.list)){
  fl <- read.csv(file.list[i], header = T)
  fl[1:5,1:10]
  nr <- nrow(fl)
  nc <- ncol(fl)
  for(j in 4:ncol(fl)){
    fl.j <- fl[,j]
    hist(fl.j)
    cl.name <- coln
    mn <- min(fl.j)
    mx <- max(fl.j)
    mmn <- mean(fl.j)
    msd <- sd(fl.j)
  }
}



################################################################################
############################## Random Forest ###################################
################################################################################

##### Set directory
setwd("~/Desktop/hs/all-csv/nrm_0-1_maxSamp-4310/")

##### List all files in directory to variable
fls <- list.files(pattern = "nrm")
length(fls)
fls

##### Get plant ID and match to treatment values
plnt.id <- t(matrix(unlist(strsplit(fls, split = "_")), nrow = 5))
plnt.id <- plnt.id[,3]
plnt.id <- gsub("SampID-", "", plnt.id)
plnt.id.trt <- cbind(fls, plnt.id)

##### Check sequence of information with match both ways
match(id.trt[,1], plnt.id.trt[,2])
match(plnt.id.trt[,2], id.trt[,1])
dim(id.trt)
dim(plnt.id.trt)

##### Sort out sample, image number, leaf, and treatment
### Split treatment files by 0,1,2
id.trt.0 <- id.trt[id.trt[,2] == 0, ]
id.trt.1 <- id.trt[id.trt[,2] == 1, ]
id.trt.2 <- id.trt[id.trt[,2] == 2, ]

### Sort out images by values matching treatment
# Get zero treatment
trt.match <- match(plnt.id.trt[,2], id.trt.0[,1])
trt.bind <- cbind(plnt.id.trt, trt.match)
trt.comp <- trt.bind[complete.cases(trt.bind[,3]), ]
trt.0 <- trt.comp
trt.0.rep <- rep(0, times = nrow(trt.0))
trt.0 <- cbind(trt.0, trt.0.rep)
dim(trt.0)

# Get one treatment
trt.match <- match(plnt.id.trt[,2], id.trt.1[,1])
trt.bind <- cbind(plnt.id.trt, trt.match)
trt.comp <- trt.bind[complete.cases(trt.bind[,3]), ]
trt.1 <- trt.comp
trt.1.rep <- rep(1, times = nrow(trt.1))
trt.1 <- cbind(trt.1, trt.1.rep)
dim(trt.1)

# Get two treatment
trt.match <- match(plnt.id.trt[,2], id.trt.2[,1])
trt.bind <- cbind(plnt.id.trt, trt.match)
trt.comp <- trt.bind[complete.cases(trt.bind[,3]), ]
trt.2 <- trt.comp
trt.2.rep <- rep(2, times = nrow(trt.2))
trt.2 <- cbind(trt.2, trt.2.rep)
dim(trt.2)

##### Combine all treatment/file names into one list
trt.all <- rbind(trt.0, trt.1, trt.2)
dim(trt.all)
trt.all[1:5,]

i <- 70
# setwd("nrm_0-1_maxSamp-4310/")
##### Combine all files
for(i in 1:nrow(trt.all)){
  fls.i <- read.csv(trt.all[i,1], header = T)
  fls.rep <- rep(as.numeric(trt.all[i,4]), times = nrow(fls.i))
  fls.bind <- cbind(fls.rep, fls.i)
  if(i == 1){
    fls.bind.p <- fls.bind
  }
  else{
    fls.bind.p <- rbind(fls.bind.p, fls.bind)
  }
}
dim(fls.bind.p)

##### Write out data
write.csv(fls.bind.p, file = "all_4310PxlSamps_norm0-1_wTreatment.csv")


################################################################################
########################### Load Pre-Processed Data ############################
################################################################################

##### Set Directory
setwd("~/Desktop/hs/all-csv/nrm_0-1_maxSamp-4310/")

##### Read in csv
fls.bind.p <- read.csv("all_4310PxlSamps_norm0-1_wTreatment.csv", header = T)
fls.bind.p <- fls.bind.p[,-1]
fls.bind.p[1:5,1:10]

rownames(fls.bind.p) <- paste(fls.bind.p[,3],fls.bind.p[,1],fls.bind.p[,5],fls.bind.p[,6], sep = "_")

mydata = fls.bind.p
mydata[1:5,1:10]
# colnames(mydata)[1:3] <- c("trt", "x_1", "x_2")
mydata = cbind(mydata[,1],mydata[,3], mydata[,7:ncol(mydata)])
colnames(mydata)[1:2] = c("trt","flnm")
mydata[1:5,1:10]
mydata <- mydata[,-2]
dim(fls.bind.p)
str(mydata)

########## Start Building actual RFM ###########
# mydata = iris

##### Split Data
# index = sample(2, nrow(mydata), replace = T, prob = c(0.05,0.95))

##### Create Testing and Training Data
set.seed(9828)
trn.tst <- mydata[sample(nrow(mydata), size = nrow(mydata), replace = F), ]


##### Split testing training by different sets of names
### Parse samples to two non-overlapping data sets
unq.nms <- unique(trn.tst[,2])
set.seed(28)
unq.nms.rnd <- sample(unq.nms, size = length(unq.nms), replace = F)
length(unq.nms.rnd)
unq.trn <- unq.nms[1:round(length(unq.nms.rnd)/2)]
unq.tst <- unq.nms[length(unq.nms):(round(length(unq.nms.rnd)/2)+1)]

### Check matches, all should be NA
match(unq.trn, unq.tst)

##### Training
# training = mydata[index==1,]
trn <- trn.tst[1:30000,]


##### Testing
# testing = mydata[index==2,]
tst <- trn.tst[nrow(trn.tst):(nrow(trn.tst)-30000), ]


##### Check testing and training dimensions
### Dimensions
dim(trn)
dim(tst)

### View in col-row format
trn[1:5,1:10]
tst[1:5,1:10]


library(randomForest)
##### Rand Forest
rfm = randomForest(trt~., data = trn)
plot(rfm)
str(rfm)
importance(rfm)
##### Evaluation
pred <- predict(rfm, tst)
tst$pred <- pred
View(tst)
str(tst)


##### Confusion Matrix
cfm <- table(tst$trt, tst$pred)
dim(cfm)
image(cfm)


##### Classification Accuracy
acc <- sum(diag(cfm)/sum(cfm))
acc


########## Create Testing and Training Data without 1 condition for RFM ##########
set.seed(5423)
trn.tst <- mydata[sample(nrow(mydata), size = nrow(mydata), replace = F), ]
trn.tst.02 <- trn.tst[trn.tst[,1] != 1, ]

##### Split testing training by different sets of names
### Parse samples to two non-overlapping data sets
unq.nms <- unique(trn.tst.02[,2])
set.seed(28)
unq.nms.rnd <- sample(unq.nms, size = length(unq.nms), replace = F)
length(unq.nms.rnd)
unq.trn <- unq.nms[1:round(length(unq.nms.rnd)/2)]
unq.tst <- unq.nms[length(unq.nms):(round(length(unq.nms.rnd)/2)+1)]

### Check matches, all should be NA
match(unq.trn, unq.tst)

##### Training
# training = mydata[index==1,]
set.seed(82)
trn.02 <- trn.tst.02[sample(nrow(trn.tst.02), size = 30000, replace = F),]
trn.02 <- trn.02[,-2]
trn.02[1:5,1:10]
dim(trn.02)

##### Testing
# testing = mydata[index==2,]
set.seed(824)
tst.02 <- trn.tst[sample(nrow(trn.tst), size = 30000, replace = F), ]
tst.02 <- tst.02[,-2]
tst.02[1:5,1:10]
dim(tst.02)

##### Check testing and training dimensions
### Dimensions
dim(trn.02)
dim(tst.02)

### View in col-row format
trn.02[1:5,1:10]
tst.02[1:5,1:10]



##### Rand Forest
rfm = randomForest(trt~., data = trn)
plot(rfm)
str(rfm)

##### Evaluation
pred <- predict(rfm, tst)
tst$pred <- pred
View(tst)
str(tst)


##### Confusion Matrix
cfm <- table(tst$trt, tst$pred)
dim(cfm)
image(cfm)


##### Classification Accuracy
acc <- sum(diag(cfm)/sum(cfm))
acc



################################################################################
####################### Decision Tree Based Prediction #########################
################################################################################

##### Create Testing Training Data
set.seed(9828)
trn.tst <- mydata[sample(nrow(mydata), size = nrow(mydata), replace = F), ]

##### Training
### Split to unique samples
trn <- trn.tst[complete.cases(match(trn.tst[,2], unq.trn)), ]
dim(trn)

### Random Order
set.seed(723)
trn <- trn[sample(nrow(trn), size = 30000, replace = F), ]
trn[1:5,1:15]
trn <- cbind(trn[,1], trn[,3:ncol(trn)])
colnames(trn)[1] <- "trt"
trn[1:5,1:15]
dim(trn)

##### Testing
tst <- trn.tst[complete.cases(match(trn.tst[,2], unq.tst)), ]
set.seed(8942)
tst <- tst[sample(nrow(tst), size = 30000, replace = F), ]
tst[1:5,1:15]
tst <- cbind(tst[,1], tst[,3:ncol(tst)])
colnames(tst)[1] <- "trt"
tst[1:5,1:15]
dim(tst)


##### Rpart Model, Decision Tree
dtm <- rpart(trt~.,trn, method = "class")
dtm
plot(dtm)
text(dtm)

rpart.plot(dtm, type = 4, extra = 101)

pred <- predict(dtm, tst, type = "class")


cfm <- table(tst[,1], pred)

##### Classification Accuracy
acc <- sum(diag(cfm)/sum(cfm))
acc


##### Decision tree without 0.5 nitrogen treatment, aka trt = 1
##### Create Testing Training Data
set.seed(9828)
trn.tst <- mydata[sample(nrow(mydata), size = nrow(mydata), replace = F), ]
trn.tst[1:5,1:7]
unique(trn.tst[,1])
trn.tst <- trn.tst[trn.tst[,1] != 1, ]
unique(trn.tst[,1])
dim(trn.tst)
trn.tst[1:5,1:7]

##### Training
### Split to unique samples
trn.02 <- trn.tst[complete.cases(match(trn.tst[,2], unq.trn)), ]
dim(trn.02)

### Random Order
set.seed(723)
trn.02 <- trn.02[sample(nrow(trn.02), size = 60000, replace = F), ]
trn.02[1:5,1:15]
trn.02 <- cbind(trn.02[,1], trn.02[,3:ncol(trn.02)])
colnames(trn.02)[1] <- "trt"
trn.02[1:5,1:15]
dim(trn.02)


##### Testing
tst.02 <- trn.tst[complete.cases(match(trn.tst[,2], unq.tst)), ]
set.seed(8942)
tst.02 <- tst.02[sample(nrow(tst.02), size = 60000, replace = F), ]
tst.02[1:5,1:15]
tst.02 <- cbind(tst.02[,1], tst.02[,3:ncol(tst.02)])
colnames(tst.02)[1] <- "trt"
tst.02[1:5,1:15]
dim(tst.02)



##### Rpart Model, Decision Tree
dtm <- rpart(trt~.,trn, method = "class")
dtm

##### Generic Plot
### R base plot
plot(dtm)
text(dtm)

### Rpart Plot
rpart.plot(dtm)

##### Check Prediction Value
pred <- predict(dtm, tst, type = "class")

##### Create Confusion Matrix
cfm <- table(tst[,1], pred)
cfm

##### Classification Accuracy
acc <- sum(diag(cfm)/sum(cfm))
acc


########## Prediction of all samples
pred.all <- predict(dtm, trn.tst, type = "class")
length(pred.all)
dim(trn.tst)
pred.nmssplt <- t(matrix(unlist(strsplit(names(pred.all), split = "_")), nrow = 6))
pred.nmssplt[1:5,]
pred.fl.nms <- paste(pred.nmssplt[,1], pred.nmssplt[,2], pred.nmssplt[,3], sep = "_")
pred.comb <- cbind(pred.all, trn.tst[,1])
pred.comb[1:5,]



##### Decision tree without 0.5 nitrogen treatment, aka trt = 1
set.seed(32)
trn.tst <- mydata[sample(nrow(mydata), size = nrow(mydata), replace = F), ]
trn.tst[1:5,1:10]

trn.tst <- trn.tst[trn.tst[,1] != 1, ]
dim(trn.tst)

##### Training
# training = mydata[index==1,]
trn <- trn.tst[1:30000,]
trn[1:5,1:15]


##### Testing
# testing = mydata[index==2,]
tst <- trn.tst[nrow(trn.tst):(nrow(trn.tst)-30000), ]
tst[1:5,1:15]



##### Rpart Model, Decision Tree
dtm <- rpart(trt~.,trn, method = "class")
dtm

##### Generic Plot
### R base plot
plot(dtm)
text(dtm)

### Rpart Plot
rpart.plot(dtm)

##### Check Prediction Value
pred <- predict(dtm, tst, type = "class")

##### Create Confusion Matrix
cfm <- table(tst[,1], pred)
cfm

##### Classification Accuracy
acc <- sum(diag(cfm)/sum(cfm))
acc





################################################################################
############### Logistic Regression Classification of High/Low #################
################################################################################

trn[trn[,1] > 0, 1] <- 1
tst[tst[,1] > 0, 1] <- 1

##### Fit the model
set.seed(723)
model <- glm(trt ~., data = trn, family = binomial)

##### Summarize the model
summary(model)


##### Make predictions
probabilities <- model %>% predict(tst, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
predicted.classes


##### Model accuracy
mean(predicted.classes == tst$X801.79nm)


################################################################################
######################### Support Vector Machine ###############################
################################################################################


# svmfit = svm(trt ~ ., data = trn, kernel = "linear", cost = 10, scale = FALSE)
# print(svmfit)

# Continue Here




################################################################################
######################### remove rownames for HS iRF ###########################
################################################################################

hs.data <- read.table("HS_for_iRF_YvectCol1_DataCol2toEnd_2022-10-20.txt")
hs.data[1:5,1:8]
hs.train <- hs.data[1:(nrow(hs.data)/2), ]
hs.pred <- hs.data[((nrow(hs.data)/2)+1):nrow(hs.data), ]
dim(hs.train)
dim(hs.pred)
hs.train[1:5,1:8]
hs.pred[1:5,1:8]

##### Three categories of N dosage
hs.train012 <- hs.train
hs.pred012 <- hs.pred

##### Two categories of N dosage, high-low
hs.train02 <- hs.train012[hs.train012[1,] != "1", ]
hs.pred02 <- hs.pred012[hs.pred012[1,] != "1", ]
hs.pred02 <- hs.pred02[,-1]
hs.pred02[1:5,1:8]

##### Two categories of N dosage, med-high
hs.train12 <- hs.train012[hs.train012[1,] != "0", ]
hs.pred12 <- hs.pred012[hs.pred012[1,] != "0", ]
hs.pred12 <- hs.pred12[,-1]
hs.pred12[1:5,1:8]

##### Two categories of N dosage, low-med
hs.train01 <- hs.train012[hs.train012[1,] != "2", ]
hs.pred01 <- hs.pred012[hs.pred012[1,] != "2", ]
hs.pred01 <- hs.pred01[,-1]
hs.pred01[1:5,1:8]

##### Remove 012 high-med-low prediction y-vector
hs.pred012 <- hs.pred012[,-1]
hs.pred012[1:5,1:8]


write.table(hs.train012, file = "HS_for_iRF_012_YvectCol1_DataCol2toEnd_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)
write.table(hs.pred012, file = "HS_for_iRF_012_NoYvect_prediction_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)
write.table(hs.train12, file = "HS_for_iRF_12_YvectCol1_DataCol2toEnd_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)
write.table(hs.pred12, file = "HS_for_iRF_12_NoYvect_prediction_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)
write.table(hs.train02, file = "HS_for_iRF_02_YvectCol1_DataCol2toEnd_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)
write.table(hs.pred02, file = "HS_for_iRF_02_NoYvect_prediction_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)
write.table(hs.train01, file = "HS_for_01_iRF_YvectCol1_DataCol2toEnd_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)
write.table(hs.pred01, file = "HS_for_iRF_01_NoYvect_prediction_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)






write.table(hs.data, file = "HS_for_iRF_YvectCol1_DataCol2toEnd_2022-11-06.txt", col.names = T, row.names = F, sep = "  ", quote = F)








################################################################################
############################# Post iRF Analysis ################################
################################################################################

##### define color plotting palette
cols <- colorRampPalette(c("grey30", "dodgerblue3", "dodgerblue1", "grey80", "indianred2", "red2"))(100)
palette(cols)


##### Set workign directory
setwd("/Users/ju0/Desktop/hs/irf_trn")

##### low-high
setwd("/Users/ju0/Desktop/hs/irf_trn/low-high/")

fls <- list.files(pattern = NULL)

fl.i <- read.table(fls[1])

dim(fl.i)


i <- 1
##### Loop through files
for(i in 1:length(fls)){
  fl.i <- read.table(fls[i])
  if(i == 1){
    fl.i.p <- fl.i[,2]
  }
  else{
    fl.i.p <- cbind(fl.i.p, fl.i[,2])
  }
}

fl.i <- rowSums(fl.i.p)
names(fl.i) <- gsub("X", "", read.table(fls[1])[,1])
dim(fl.i.p)

##### Get Mean and Standard Dev.
fls.sd <- apply(fl.i.p, 1, sd)
fls.mn <- apply(fl.i.p, 1, mean)

##### Set Names
names(fls.sd) <- names(fl.i)
names(fls.mn) <- names(fl.i)

par(mfrow = c(1,1))
barplot(fls.mn, col = "grey", las = 2, cex.names = 0.5)
barplot(fls.sd, las = 2, cex.names = 0.5, col = fls.sd)

##### Top20
fls.mn.srt <- sort(fls.mn, decreasing = T)
fls.sd.srt <- sort(fls.sd, decreasing = T)
barplot(fls.mn.srt[1:20], col = "grey", las = 2, cex.names = 1)

##### Top 20 with stdev
fls.mn.sd <- cbind(fls.mn, fls.sd)
dim(fls.mn.sd)
fls.mn.sd <- fls.mn.sd[order(fls.mn.sd[,1], decreasing = T), ]


barplot(fls.mn.srt[1:20], col = "grey", las = 2, cex.names = 1, border = F)
error.bar <- function(x, y, upper, lower=upper, length=0.1,...){
  arrows(x, y+upper, x, y-lower, angle = 90, code = 3, length=length, col = "grey30", ...)
}


##### Plot iRF outputs top 20 Contributing Features
ze_barplot <- barplot(fls.mn.sd[1:20,1], col = "dodgerblue1", las = 2, cex.names = 0.8, border = F, ylim = c(0, 1.2*max(fls.mn.sd[1:20,1])))
error.bar(ze_barplot, fls.mn.sd[1:20,1], fls.mn.sd[1:20,2])




##### Plot Results
par(mfrow = c(1,1))
barplot(fls.mn, names = gsub(":","",gsub("X","",names(fl.i))), las = 2, cex.names = 0.5,
        col = round((((fls.mn-min(fls.mn))/max(fls.mn))+0.01)*100), main = "Importance Values of iRF on All 580k, 290k/290k Test/Pred Split HS Leaf Pixels")


spltwts <- fls.sd
barplot(as.numeric(spltwts), names = gsub(":","",gsub("X","",names(fl.i))), las = 2, cex.names = 0.5,
        col = round((((fls.sd-min(fls.sd))/max(fls.sd))+0.01)*100), main = "Split Weights of iRF on All 581,851 HS Leaf Pixels")




################################################################################
#################### Keras Based CNN classifier draft ##########################
################################################################################


#### Load packages
library(keras)
library(tidyverse)

##### Read in data sets
my_data <- read.csv("mydata.csv")
my_data <-

##### Split training and Prediction
train_data <- my_data[1:800,]
test_data <- my_data[801:1000,]

##### Set up y-vector data
y_vector <- train_data[,1]

##### Set scaling to 0-1
train_data_scaled <- scale(train_data[,-1])
test_data_scaled <- scale(test_data[,-1])

##### Name cnn model as model
model <- keras_model_sequential()

##### Add layers to the model
model %>%
  layer_dense(units = 8, activation = "relu", input_shape = ncol(train_data_scaled)) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")

##### Compile Model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "Adam",
  metrics = c("accuracy")
)

###### Fit cnn model, set epochs and batch size
fit_model <- model %>% fit(
  train_data_scaled,
  y_vector,
  epochs = 10,
  batch_size = 32
)

##### Evaluate cnn model
test_results <- model %>% evaluate(
  test_data_scaled,
  y_vector
)

##### Print accuracy
cat("Test Accuracy = ", test_results[[1]], "\n")

##### Plot accuracy
plot(fit_model)

##### Plot Entropy Loss
plot(fit_model, plot = "loss")
