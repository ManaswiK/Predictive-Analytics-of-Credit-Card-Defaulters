We developed this project in R language

The installation statements for the packages used are

#install.packages('randomForest')
#install.packages("rpart", dependencies=TRUE)
#install.packages('ipred')
#install.packages('ada')
#install.packages('class')
#install.packages('gbm')
#install.packages("neuralnet",dependencies=TRUE, repos="http://cran.us.r-project.org")
#install.packages('e1071', dependencies = TRUE, repos="http://cran.us.r-project.org")

Statements to load the packages are

library(e1071)
library(neuralnet)
library(rpart)
library(randomForest)
library(ipred)
library(caret)
library(ada)
library(gbm)
library(class)

We have used the "default of credit card clients" dataset

Link to the dataset is:
http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

We have attached the ".csv" file of the dataset

Note: The paths to the datasets are hard coded in the program. Hence the path has to be specified/changed in the code.
eg:They are given as "C:/Users/manwi/creditcard.csv" in the code, So be sure to change the path.

Contributions of the team members:

Pratyusha Reddy Velama: Developed four classifiers SVM, Gradient boosting, kNN and NaiveBayes Classifier
Manwitha Yarradoddi: Developed three classifiers Decision tree, Adaboost, Neural networks
Manaswi Kothi: Developed three classifiers Random Forest, perceptron and bagging

Report: Done by all three
