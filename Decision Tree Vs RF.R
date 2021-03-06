library(caret)# For K fold cross validation and accuracy plot
library(rpart)# For modelling decison tree
library(FSelector)# For calculating IG
library(rpart.plot)# plotting an tree
library(randomForest)# For modelling random forest
library(XLConnect)# For reading an Illness excel
library(readxl)# For reading an Illness excel
library("ROCR")#For plotting ROC curve

getwd()
Illness = read_xlsx("illness.xlsx", col_names = TRUE)
Illness$test_result = as.factor(Illness$test_result)

#Splitting the data into test, train and validation data set
set.seed(99)
ind = sample(2, size = nrow(Illness), prob = c(0.7,0.3), replace = TRUE)
Train_data = Illness[ind==1,] 
Test_data = Illness[ind==2,]


#Calculating Information Gain
IG = information.gain(test_result~., data = Train_data, unit = "log2")
#                   attr_importance
# plasma_glucose       0.15150770
# bp                   0.00000000
# skin_thickness       0.00000000
# num_pregnancies      0.00000000
# insulin              0.11023022
# bmi                  0.06990622
# pedigree             0.00000000
# age                  0.07741410




#**************************************************************************************
#                       MODEL 1: Decision Tree
#**************************************************************************************

#Decision Tree
D_Tree = rpart(test_result~., data = Train_data, method = "class")
rpart.plot(D_Tree)



#Predicting the test data set and finding the accuracy
predict_D_Tree = predict(D_Tree, Test_data, type = "class")
Missclassification_error = mean(predict_D_Tree != Test_data$test_result) #Error is 29.7%
Missclassification_table = table(prediction = predict_D_Tree, Actual = Test_data$test_result)
confusionMatrix(Missclassification_table)
#Accuracy on testing data is 70.25% without k fold validation


#Pruning the tree:(Tuning Parameter)
printcp(D_Tree)
plotcp(D_Tree)

cp = D_Tree$cptable[which.min(D_Tree$cptable[,"xerror"]),"CP"]
pruned_D_Tree = prune(D_Tree, cp = cp) #minimum cp = 0.05405405
rpart.plot(pruned_D_Tree)

predict_PD_Tree = predict(pruned_D_Tree, Test_data, type = "class")
Missclassification_error = mean(predict_PD_Tree != Test_data$test_result) #Error is 19%
Missclassification_table = table(prediction = predict_PD_Tree, Actual = Test_data$test_result)
confusionMatrix(Missclassification_table)
#Accuracy of pruned tree on testing data set is 80.17%


#K = 10 fold cross validation
#Modelling decsion tree using 10 fold cv on testing)
control <- trainControl(method="repeatedcv", number=10, savePredictions = TRUE, repeats = 4, search = "random")
D_Tree_CV_model <- train(test_result~., data=Train_data, method="rpart", trControl=control, metric = "Accuracy", tuneLength = 15)
print(D_Tree_CV_model)
plot(D_Tree_CV_model)


#Testing with Test data set:
predict_KD_Tree = predict(D_Tree_CV_model, Test_data, method = "class")
Missclassification_table = table(prediction = predict_KD_Tree, Actual = Test_data$test_result)
confusionMatrix(Missclassification_table)
#Accuracy when 10 fold cv for a model on validation set is 80.17% which is same as pruned tree accuracy





#******************************************************************************************************************
#                 MODEL 2: Random Forest
#******************************************************************************************************************

RF_Tree = randomForest(test_result~., data = Train_data)#OOB error: 25.1%
#default value of mtry is sqrt of no of attributes ie 3
varImpPlot(RF_Tree)

#Predicting the test data set and finding the accuracy
predict_RF_Tree = predict(RF_Tree, Test_data, method = "class")
Missclassification_error = mean(predict_RF_Tree != Test_data$test_result) 
Missclassification_table = table(prediction = predict_RF_Tree, Actual = Test_data$test_result)
confusionMatrix(Missclassification_table)
#Accuracy on testing data is 80.99%


#Manual tuning: Using tuneRF function to find best mtry:
bestmtry = tuneRF(Train_data, Train_data$test_result, stepFactor = 1.2, improve = 0.01, trace = T, plot = T)
#best mtry is = 3
RF_Tree_MT = randomForest(test_result~., data = Train_data, mtr = 3)
predict_RF_Tree_M = predict(RF_Tree_MT, Test_data, method = "class")
Missclassification_error = mean(predict_RF_Tree_M != Test_data$test_result) 
Missclassification_table = table(prediction = predict_RF_Tree_M, Actual = Test_data$test_result)
confusionMatrix(Missclassification_table)
#Manual Tuning accuracy = 81%


#Auto tuning for mtry
control <- trainControl(method="repeatedcv", number=10, repeats=2, savePredictions = TRUE, search = "random")
metric = "Accuracy"
RF_Tree_CV <- train(test_result~., data=Train_data, method="rf", metric=metric, tuneLength=5, trControl=control)
print(RF_Tree_CV)
plot(RF_Tree_CV)


#Testing with test data set:
predict_KRF_Tree = predict(RF_Tree_CV, Test_data, method = "class")
Missclassification_table = table(prediction = predict_KRF_Tree, Actual = Test_data$test_result)
confusionMatrix(Missclassification_table)
#Accuracy when k =10 cv for a model on validation set is 
#Accuracy     mtry
#81.82        3





#*********************************************************************************************************************************
#                     ROC for Random Forest and Decision Tree
#*********************************************************************************************************************************

#ROC curve tpr vs fpr and values for different cutoff values
#Decison Tree
ROC_Data_DT = predict(D_Tree_CV_model,Test_data, type = 'prob')
ROC_fpr_tpr_DT = prediction(ROC_Data_DT[,2], Test_data$test_result)
auc_DT = performance(ROC_fpr_tpr_DT, measure = 'auc')
(Area_under_curve_DT = auc_DT@y.values[[1]])
perf_DT = performance(ROC_fpr_tpr_DT, 'tpr','fpr')


cutoffs_DT <- data.frame(cut=perf_DT@alpha.values[[1]], fpr=perf_DT@x.values[[1]], 
                         tpr=perf_DT@y.values[[1]])
best_cutoff_DT = cutoffs_DT[which.max(cutoffs_DT$tpr + (1- cutoffs_DT$fpr)), ]

plot(performance(ROC_fpr_tpr_DT, 'tpr','fpr'), colorize = TRUE, main = "ROC Curve Decion Tree Model", sub = bquote("AUROC="~.(Area_under_curve_DT)))
abline(coef = c(0,1))
abline(v = best_cutoff_DT$fpr, h = best_cutoff_DT$tpr, lty = 3, col = "Red", pch = 0.2)
text(x = best_cutoff_DT$fpr, y = (best_cutoff_DT$tpr+0.08), labels ="Cut off Value")

#Random Forest

ROC_Data = predict(RF_Tree_CV,Test_data, type = 'prob')
ROC_fpr_tpr = prediction(ROC_Data[,2], Test_data$test_result)
auc = performance(ROC_fpr_tpr, measure = 'auc')
(Area_under_curve = auc@y.values[[1]])
perf_RF = performance(ROC_fpr_tpr, "tpr","fpr")

cutoffs_RF <- data.frame(cut=perf_RF@alpha.values[[1]], fpr=perf_RF@x.values[[1]], tpr=perf_RF@y.values[[1]])
best_cutoff_RF = cutoffs_RF[which.max(cutoffs_RF$tpr + (1- cutoffs_RF$fpr)), ]

plot(performance(ROC_fpr_tpr, "tpr","fpr"), colorize = TRUE, main = "ROC Curve Random Forest", sub = bquote("AUROC="~.(Area_under_curve)))
abline(coef = c(0,1), lty = 2)
abline(v = best_cutoff_RF$fpr, h = best_cutoff_RF$tpr, lty = 3, col = "Red", pch = 0.2)
text(x = best_cutoff_RF$fpr, y = (best_cutoff_RF$tpr+0.08), labels ="Cut off Value")

#ROC curve in single plot for Decision Tree and Random Forest

plot(performance(ROC_fpr_tpr, "tpr","fpr"), col = 'Green', main = "ROC Curve DT and RF")
par(new = TRUE)
plot(performance(ROC_fpr_tpr_DT, 'tpr','fpr'), col = 'Blue', axes = FALSE)
abline(coef = c(0,1))
legend("bottomright", legend = c("Random Forest", "Decision Tree"), col = c('Green', 'Blue'), lty = 1.2, cex = 0.8)






#********************************************************************************************************************************
#                               LEARNING CURVE for Decision Tree and Random Forest
#********************************************************************************************************************************

#Decision Tree Learning Curve
x = c(10,40,90,150,200,245)
learning_data = data.frame(Train_size = numeric(), Accuracy = numeric(), Model = character(), stringsAsFactors = FALSE)
DT_Learning = data.frame(Train_size = numeric(), Accuracy = numeric(), Model = character(), stringsAsFactors = FALSE)
s = 0
for(i in x)
{
  
  control <- trainControl(method="repeatedcv", number=10, savePredictions = TRUE, search = "random")
  metric = "Accuracy"
  RF_Tree_CV <- train(test_result~., data=Train_data[1:i,], method="rf", metric=metric, tuneLength=5, trControl=control)
  
  predict_KRF_Tree = predict(RF_Tree_CV, Test_data, method = "class")
  Missclassification_table = table(prediction = predict_KRF_Tree, Actual = Test_data$test_result)
  accur = confusionMatrix(Missclassification_table)$overall[[1]]
  
  s = s+1
  learning_data[s,1] = i
  learning_data[s,2] = accur
  learning_data[s,3] = "Random Forest"
  
#Dtree Learning Curve
  
  control <- trainControl(method="repeatedcv", number=10, savePredictions = TRUE, search = "random")
  D_Tree_CV_model <- train(test_result~., data=Train_data[1:i,], method="rpart", trControl=control, metric = "Accuracy", tuneLength = 5)

  
  predict_KD_Tree = predict(D_Tree_CV_model, Test_data, method = "class")
  Missclassification_table = table(prediction = predict_KD_Tree, Actual = Test_data$test_result)
  dt_acur = confusionMatrix(Missclassification_table)$overall[[1]]
  
  
  DT_Learning[s,1] = i
  DT_Learning[s,2] = dt_acur
  DT_Learning[s,3] = "Decision Tree" 
}
T_learning_data = rbind(learning_data, DT_Learning)
T_learning_data$Model = as.factor(T_learning_data$Model)

ggplot(T_learning_data)+geom_point(mapping = aes(x = Train_size, y = Accuracy, color = Model))+geom_smooth(mapping = aes(x = Train_size, y = Accuracy, color = Model), se = F)+
labs(title = "Learning Curve",y = "Accuracy (against test set)" )+ theme(plot.title = element_text(hjust = 0.5))

