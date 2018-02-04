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

slug <- na.omit(as.data.frame(read_csv("assesment_Joe2.csv")))




nzv <- nearZeroVar(slug,
                   saveMetrics = TRUE)
cutOff= .03

colz <- c(rownames(nzv[nzv$percentUnique > cutOff,]))

completeData <-
  as.data.frame(slug[,colz])

set.seed(601)
trainIndex <- createDataPartition(y=completeData$Column5,
                                   p = .70,
                                   list = FALSE)

training = as.data.frame(completeData[trainIndex,])

testing = as.data.frame(completeData[-trainIndex,])


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


#---------------------------------------------------------
trainingN = as.factor(training)
testingN = as.factor(testing)

c50MOD <- train(Column5 ~ ., data = trainingN,
                                           method = "C5.0",
                                           preProcess = c("center","scale"))
save(c50MOD, file = "c50MOD.rda")

#prediction
c50_predict <- predict(c50MOD, testingN)
#@Capture metrics####
c50_summary <- capture.output(c50MOD)
cat("Summary", svm_summary,
    file = "summary of gbm_summary.txt",
    sep = "\n",
    append = TRUE)
c50MOD

postResample(c50_predict, trainingN$Column5)

#---------------------------Summary----------------------------------------

knn =as.data.frame(postResample(knn_predict, testing$Column5))
rf = as.data.frame(postResample(rf_predict, testing$Column5))
svm = as.data.frame(postResample(svm_predict, testing$Column5))
gbm = as.data.frame(postResample(gbm_predict, testing$Column5))

results = data.frame(knn,rf,svm,gbm)
results #----------------------------> KNN Wins


