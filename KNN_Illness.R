library(caret)# For K fold cross validation and accuracy plot
library(rpart)# For modelling decison tree
library(FSelector)# For calculating IG
library(rpart.plot)# plotting an tree
library(randomForest)# For modelling random forest
library(XLConnect)# For reading an Illness excel
library(readxl)# For reading an Illness excel
library("class")#For KNN

#Information gain:
#IG = information.gain(test_result~., data = Train_data, unit = "log2")

getwd()
Illness = read_xlsx("illness.xlsx", col_names = TRUE)
Illness$test_result = as.factor(Illness$test_result)


#Splitting the data into test, train data set
set.seed(99)
ind = sample(2, size = nrow(Illness), prob = c(0.7,0.3), replace = TRUE)
#Z-normalization
z_norm = function(x){((x-min(x))/(max(x) - min(x)))}
norm_illness = as.data.frame(lapply(Illness[,-3], z_norm))
Train_data = norm_illness[ind==1,]
Test_data = norm_illness[ind==2,]
Train_tr_label = Illness$test_result[ind==1]
Test_tr_label = Illness$test_result[ind==2]

cv_train_data = as.data.frame(Illness[ind==1,])
cv_test_data = as.data.frame(Illness[ind==2,])
  
summary(norm_illness)#Summary before normalization
summary(Illness)#Summary after normalization



knn_model = knn(Train_data, Test_data, Train_tr_label, k=7)
table(predicted = knn_model,actual = Test_tr_label)
confusionMatrix(table(predicted = knn_model,actual = Test_tr_label))
#Accuracy on test data for k=7 is 0.7586


#*************  10-Fold Cross Validation and best K(neighbours) ******************************

#10-fold validation using method = "knn"
train_ctl = trainControl(method="repeatedcv", number=10, repeats=3)
Knn_model_cv = train(x = cv_train_data[,-3], y = cv_train_data$test_result, metric = "Accuracy", method = "knn", tuneGrid = expand.grid(k=c(1:25)), trControl = train_ctl, preProcess = "scale")
plot(Knn_model_cv)

knnPredict_cv <- predict(Knn_model_cv,cv_test_data)
confusionMatrix(table(knnPredict_cv, cv_test_data$test_result))

#Selected k = 19. Accuracy on training data is 77.84%
#For k=19, accuracy on testing data is: 77.59%

#Ploting ROC and AUC curves for knn:
library("pROC")
ROC_Data = predict(Knn_model_cv,cv_test_data, type = 'prob')
auc_knn = auc(cv_test_data$test_result, ROC_Data[,2])#81.35
plot(roc(cv_test_data$test_result, ROC_Data[,2]))

#************************** Inverse distance KNN (KKNN)******************************************

#10-fold validation using method = kknn
train_ctl1 = trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = "all")
kknn_model = train(x = cv_train_data[,-3], y = cv_train_data$test_result, metric = "Accuracy", method = "kknn", trControl = train_ctl, preProcess = "scale")
plot(kknn_model)
#kmax = 5, distance = 2 and kernel = optimal. Accuracy on train data is: 76.822%

kknnPredict <- predict(kknn_model,cv_test_data)
confusionMatrix(table(kknnPredict, cv_test_data$test_result))
#Accuracy on test data is : 73.28%


#Ploting ROC and AUC curves for kknn:
kknn_ROC_Data = predict(kknn_model,cv_test_data, type = 'prob')
auc_kknn = auc(cv_test_data$test_result, kknn_ROC_Data[,2]) #74.65
plot(roc(cv_test_data$test_result, kknn_ROC_Data[,2]))
