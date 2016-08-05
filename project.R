#install.packages('randomForest')
#install.packages("rpart", dependencies=TRUE)
#install.packages('ipred')
#install.packages('adabag')
#install.packages('ada')
#install.packages('class')
#install.packages('gbm')
#install.packages("rpart", dependencies=TRUE)
#install.packages("neuralnet",dependencies=TRUE, repos="http://cran.us.r-project.org")
#install.packages('e1071', dependencies = TRUE, repos="http://cran.us.r-project.org")
#install.packages('ROCR')
#install.packages('ggplot2')
require('ROCR')
library('ggplot2')
library(ROCR)
library(e1071)
library(neuralnet)
library(rpart)
library(randomForest)
library(rpart)
library(ipred)
library(adabag)
library(caret)
library(ada)
library(gbm)
library(class)
createModel <- function(method,trainingData,testData, Class, ClassIndex)
{
  eval <- c(0,0,0,0)
  if(method == "decisionTree")
  {
    cl<-trainingData[,ClassIndex]
    Tree<-rpart(cl ~ .,data=trainingData,method='class',
                parms = list(split = 'information'),na.action=na.omit)
    predicted<-predict(Tree,testData,type='class')
    dt_table <- table(testData[,ClassIndex], predicted)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
  }
  if(method == "NaiveBayes")
  {
    #trainingData<-as.data.frame(trainingData)
    #testData<-as.data.frame(testData)
    cl<-trainingData[,ClassIndex]
    NB <-naiveBayes(as.factor(cl)~.,data=trainingData, na.action=na.omit)
    predicted<-predict(NB,testData,type='class')
    dt_table <- table(testData[,ClassIndex], predicted)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
    #print(table(predictedClass,testData[,ClassIndex]))
  #  accuracy<- (sum(predictedClass==testData[,ClassIndex]))/length(testData[,ClassIndex])*100.0
    #print(paste(a," NB Accuracy is ", accuracy))
    #print(accuracy)
  #  return(accuracy)
  }
  if(method == "svmClassification")
  {
    
    cl<-trainingData[,ClassIndex]
    svmModel <- svm(as.factor(cl) ~ ., data = trainingData, kernel='linear', na.action=na.omit)
    predicted<-predict(svmModel,testData,type='class')
    dt_table <- table(testData[,ClassIndex], predicted)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
  }  
  if(method== "perceptron")
  {
    n <- names(trainingData)
    fmla <- as.formula(paste(Class," ~", paste(n[!n %in% c('V35',Class)], collapse = " + ")))
    if(ClassIndex !=35 && ncol(trainingData) == 35)
      mylogit <- glm(fmla, data = trainingData[,-35], family = "binomial")
    else
      mylogit <- glm(fmla, data = trainingData, family = "binomial")
    if(ClassIndex != 35 && ncol(trainingData)==35)
    {
      prediction <- predict(mylogit, testData[,-35],type="link")
    }
    else
    {
      prediction <- predict(mylogit, testData,type="link")
    }
    #df = data.frame()
    df = ifelse(prediction>=0.5, 1 , 0)
  #  accuracy <-  df == testData[,ClassIndex] 
    #print(paste("Perceptron accuracy is",mean(accuracy)*100.0))
  #  return(mean(accuracy)*100.0)
    
  #  predicted<-predict(svmModel,testData,type='class')
    dt_table <- table(testData[,ClassIndex], df)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
  }
  if(method == "NeuralNetworks")
  {
    n <- names(trainingData)
    fmla <- as.formula(paste(Class," ~", paste(n[!n %in% Class], collapse = " + ")))
    model <- neuralnet(fmla, trainingData, hidden = 4, lifesign = "minimal", 
                       linear.output = FALSE, threshold = 0.1)
    model.results <- compute(model, testData[,-ClassIndex])
    model.results$net.result <- round(model.results$net.result)
    #accuracy <-  model.results$net.result == testData[,ClassIndex] 
    
    dt_table <- table(testData[,ClassIndex], model.results$net.result)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
    #print(paste("the accuracy of Neural Network is",mean(accuracy)*100.0))
    #return(mean(accuracy)*100.0)
  }
  
  if(method == "randomForest")
  {
    cl<-trainingData[,ClassIndex]
    set.seed(415)
    rf <- randomForest(as.factor(cl) ~ .,data=trainingData,importance=TRUE, ntree=2000)
    predicted<-predict(rf,testData)
    dt_table <- table(testData[,ClassIndex], predicted)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
  }
  if(method == "bagging")
  {
    
    fmla <- as.formula(paste("as.factor(",Class,") ~","." ))
    bag <- ipred::bagging(fmla, data=trainingData, boos = TRUE,mfinal=10,
                          control = rpart.control(cp = 0)) 
    predicted<-predict(bag,testData)
    dt_table <- table(testData[,ClassIndex], predicted)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
  }
  if(method == "adaboosting")
  {
    n <- names(trainingData)
    fmla <- as.formula(paste(Class," ~", paste(n[!n %in% Class], collapse = " + ")))
    adaboost <- ada(fmla, data = trainingData, iter=20, nu=1, type="discrete")
    predicted<-predict(adaboost,testData)
    dt_table <- table(testData[,ClassIndex], predicted)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
  }
  
  if(method == "gbmboosting")
  {
    gb = gbm.fit(trainingData[,1:(ClassIndex-1)],trainingData[,ClassIndex],n.trees=1,verbose = FALSE,shrinkage=0.001 ,bag.fraction = 0.3 ,interaction.depth = 1,
                 n.minobsinnode = 1, distribution = "bernoulli")  
    predicted <- predict(gb,testData[,1:(ClassIndex-1)],n.trees=1)
    dt_table <- table(testData[,ClassIndex], predicted)
    accuracy <- (sum(diag(dt_table)) / sum(dt_table))*100.0
    precision <- diag(dt_table) / rowSums(dt_table)
    precision <- mean(precision)
    recall <- (diag(dt_table) / colSums(dt_table))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- accuracy
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
  }
  if(method == "knn")
  {
    responseY <- as.matrix(trainingData[,dim(trainingData)[2]])
    model.knn <- knn(train= trainingData, test=testData, cl=responseY, k=k, prob=T)
    responseY1 <- as.matrix(testData[,dim(testData)[2]])
    table1 <- table(model.knn,responseY1)
    totalaccuracy =0
    accuracy =0
    for(i in (1:nrow(table1)))
    {
      for(j in (1:ncol(table1)))
      {
        if(i == j)
        {
          accuracy = accuracy + table1[i,j]
        }
        totalaccuracy = totalaccuracy + table1[i,j]
      }
    }
    efficiency = (accuracy/totalaccuracy)
    precision <- diag(table1) / rowSums(table1)
    precision <- mean(precision)
    recall <- (diag(table1) / colSums(table1))
    recall <- mean(recall)
    FScore <- (2*precision*recall)/(precision+recall)
    FScore <- mean(FScore)
    eval[1] <- mean(efficiency)*100.0
    eval[2] <- precision
    eval[3] <- recall
    eval[4] <- FScore
    return(eval)
    
  }
  
}
dataURL <- c("C:/Users/manwi/creditcard.csv")
for(p in dataURL)
{
  d<-read.csv(p,header = T,skip = 1)
  print(paste("no of instances",nrow(d)))
  print(paste("no of attributes",ncol(d)-1))
  Class <- colnames(d)[length(d)]
  ClassIndex <- length(d)
  n=nrow(d)
  k=10 #Folds
  sum = 0
  id <- sample(1:k,nrow(d),replace=TRUE)
  list <- 1:k
  A=c("decisionTree","NaiveBayes","svmClassification","perceptron","NeuralNetworks","randomForest","knn","adaboosting","bagging","gbmboosting")
  #A=c("decisionTree")
  #A=c("svmClassification","NaiveBayes","bagging","gbmboosting")
  for(a in A)
  {
    sum_val <- c(0,0,0,0)
    for (i in 1:k)
    {
    trainingData <- subset(d, id %in% list[-i])
    testData <- subset(d, id %in% c(i))
    val <- createModel(a, trainingData = trainingData,testData = testData, Class, ClassIndex)
    sum_val[1] <- sum_val[1] + val[1]
    sum_val[2] <- sum_val[2] + val[2]
    sum_val[3] <- sum_val[3] + val[3]
    sum_val[4] <- sum_val[4] + val[4]
  
    }
    print(paste("**********",a,"**********"))
  acc <- sum_val[1]/k
  print(paste(a," Accuracy is ", acc))
  precision <- sum_val[2]/k
  print(paste(a," Precision is ", precision))
  recall <- sum_val[3]/k
  print(paste(a," Recall is ", recall))
  FScore <- sum_val[4]/k
  print(paste(a," FScore is ", FScore))
  }
}
