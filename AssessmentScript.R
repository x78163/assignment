library(ipft)
library(readr)
library(caret)
library(e1071)
library(C50)
library(scatterplot3d)
library(plotly)
library(stats)
library(FactoMineR)
library(Amelia)
library(factoextra)
library(GGally)
library(mlbench)
library(PerformanceAnalytics)
library(dplyr)#for preprocessing data
library(kknn)#ML
library(ModelMetrics)#for the metrics
library(randomForest)#ML
library(foreach)
library(MLmetrics)

# Import the data slug as numeric
slug <- na.omit(as.data.frame(read_csv("assesment_Joe2.csv")))

# Begin parallel transformation of data slug into factors
slugF1 = as.factor(slug$Column1)
slugF2 = as.factor(slug$Column2)
slugF3 = as.factor(slug$Column3)
slugF4 = as.factor(slug$Column4)
slugF5 = as.factor(slug$Column5)
factorSlug = data.frame(slugF1,slugF2,slugF3,slugF4, slugF5)
colnames(factorSlug) = c("Column1", "Column2", "Column3","Column4", "Column5")
#check for near zero variance

nzv <- nearZeroVar(slug,
                   saveMetrics = TRUE)

#define cutoff for NZV
cutOff= .03

#build registry list for NZV based on cutoff parameters
colz <- c(rownames(nzv[nzv$percentUnique > cutOff,]))

#apply cutoff for NZV on data slug
completeData <-
  as.data.frame(slug[,colz])


#Setting seed for replicability
set.seed(601)

#building 70/30 training and testing partition index
trainIndex <- createDataPartition(y=completeData$Column5,
                                   p = .70,
                                   list = FALSE)

trainIndexFactor <- createDataPartition(y=factorSlug$Column5,
                                  p = .70,
                                  list = FALSE)

#applying training/testing partition index to slug
training = as.data.frame(completeData[trainIndex,])
testing = as.data.frame(completeData[-trainIndex,])

#applying training/testing partition index to factorSlug
trainingFactor= as.data.frame(factorSlug[trainIndexFactor,])
testingFactor= as.data.frame(factorSlug[-trainIndexFactor,])


#Running KNN Model and applying to testing dataframe
knnMod <- train.kknn(Column5~.,
                      data = training,
                      trControl = ctrl,
                      method = "kknn",
                      method = "optimal",
                      kmax = 5)

save(knnMod, file = "knnMod.rda")

#prediction
knn_predict <- predict(knnMod, testing)
#@Capture metrics####
knn_summary <- capture.output(knnMod)
cat("Summary", knn_summary,
    file = "summary of kknn_LO.txt",
    sep = "\n",
    append = TRUE)
knnMod

postResample(knn_predict, testing$Column5)

#__________________________________________________________________________________
#Running RF Model and applying to testing dataframe
rfMOD <- train(Column5~.,
                      data = training,
                      method ='rf',
                   
                      importance = TRUE,
                      ntree= 500,
                      maximize =TRUE
)

save(rfMOD, file = "rfMOD.rda")

#prediction
rf_predict <- predict(rfMOD, testing)
#@Capture metrics####
rf_summary <- capture.output(rfMOD)
cat("Summary", rf_summary,
    file = "summary of rf_summary.txt",
    sep = "\n",
    append = TRUE)
rfMOD

postResample(rf_predict, testing$Column5)

#----------------------------------------------------------------------------------------
#Running SVM Model and applying to testing dataframe
svmMOD <- train(Column5 ~ ., data = training,
                method = "svmLinear2",
             
              preProc = c("center", "scale")
)

save(svmMOD, file = "svmMOD.rda")

#prediction
svm_predict <- predict(svmMOD, testing)
#@Capture metrics####
svm_summary <- capture.output(svmMOD)
cat("Summary", svm_summary,
    file = "summary of svm_summary.txt",
    sep = "\n",
    append = TRUE)
svmMOD

postResample(svm_predict, testing$Column5)

#--------------------------------------------------------------------------------
#Running gbm Model and applying to testing dataframe
gbmMOD = train(Column5 ~ ., data = training,
                     method = "gbm", 
                     preProcess = c("center","scale") )

save(gbmMOD, file = "sgbmMOD.rda")

#prediction
gbm_predict <- predict(gbmMOD, testing)
#@Capture metrics####
gbm_summary <- capture.output(gbmMOD)
cat("Summary", svm_summary,
    file = "summary of gbm_summary.txt",
    sep = "\n",
    append = TRUE)
gbmMOD

postResample(gbm_predict, testing$Column5)


#-----------------C50 model, Using factorSlug, the data slug converted to factors ----------------------------------------



c50MOD <- train(Column5 ~ ., data = trainingFactor,
                                           method = "C5.0",
                                           preProcess = c("center","scale"))
save(c50MOD, file = "c50MOD.rda")

#prediction
c50_predict <- predict(c50MOD, testingFactor)
#@Capture metrics####
c50_summary <- capture.output(c50MOD)
cat("Summary", svm_summary,
    file = "summary of c50_summary.txt",
    sep = "\n",
    append = TRUE)
c50MOD

postResample(c50_predict, testingFactor$Column5)
# Accuracy     Kappa 
# 0.9421864 0.9375921 

#-----------------C50 model, Using factorSlug, the data slug converted to factors ----------------------------------------
nbMOD <- train(Column5 ~ ., data = trainingFactor,
                                           method = "nb",  
                                           preProcess = c("center","scale"))



save(nbMOD, file = "nbMOD.rda")

#prediction
nb_predict <- predict(nbMOD, testingFactor)
#@Capture metrics####
nb_summary <- capture.output(nbMOD)
cat("Summary", svm_summary,
    file = "summary of nb_summary.txt",
    sep = "\n",
    append = TRUE)
nbMOD

postResample(nb_predict, testingFactor$Column5)
# Accuracy     Kappa 
# 0.1608269 0.0000000 

# -----> Best accuracy is .94, which is beaten by the regression models at .99

#---------------------------Summary----------------------------------------

#adding postResample results to separate dataframes
knn =as.data.frame(postResample(knn_predict, testing$Column5))
rf = as.data.frame(postResample(rf_predict, testing$Column5))
svm = as.data.frame(postResample(svm_predict, testing$Column5))
gbm = as.data.frame(postResample(gbm_predict, testing$Column5))
c50 = as.data.frame(postResample(c50_predict, testingFactor$Column5))
nb = as.data.frame(postResample(nb_predict, testingFactor$Column5))

#concatenating postResample results into results dataframe for comparison
results = data.frame(knn,rf,svm,gbm)
colnames(results)= c("knn", "rf", "svm", "gbm")
results #----------------------------> KNN Wins

#              knn        rf        svm       gbm
# RMSE     0.2811031 0.6717396 6.04734457 1.2232427
# Rsquared 0.9972510 0.9915948 0.02167892 0.9587368
# MAE      0.1141705 0.3668446 3.47682702 0.8898913

# run model on the data slug, round to nearest integer and then convert to integer (important in that order)
knn_predictFinal <- as.integer(round(predict(knnMod, slug), digits = 1))

#create data table for output 
output = slug

#write new column of predictions
output$predictKNN = knn_predictFinal


#execute post resample using predicted values against column 5 of original data slug
postResample(output$predictKNN, slug$Column5)

#      RMSE  Rsquared       MAE 
# 0.5025766 0.9927981 0.2402672 

#write CSV file with the prediction column
write.csv( output, "Col5Predictions.csv")

#---> performance metrics are better if no rounding is conducted for values.  However, a post resample of 'float' variables against 'integers'
# provides an inaccurate results.  1.1 is not 1.  Since all column 5 values are integers, all results must be integers.
#--> Both correlation and regression algorithms were run on the data.  Regression algorithms proved more accurate and are better at predicting values
# Therefore final analysis only included the regression models.  
