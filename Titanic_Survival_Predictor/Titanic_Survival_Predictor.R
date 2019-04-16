library (xgboost)
library (magrittr)
library (dplyr)
library (Matrix)
library(tidyselect)
library(keras)
library(tidyverse)
library(recipes)
library(ROCR)
library(mlbench)
library(DataExplorer)
library(tidyverse)
library(polycor)
library(car)
library(broom)
library(dplyr)
library(rsample)
library(class)
library(caret)
library(ROSE)
library(randomForest)
library(mlbench)
library(caret)

raw_data<-Titanic_Full%>%select(-name)%>%select(-body)%>%select(-ticket)%>%select(-home.dest)%>%select(-cabin)%>%select(-boat)
raw_data<-raw_data%>%select(survived,everything())
plot_missing(Titanic_Full)
raw_data[,"survived"]<-as.factor(raw_data[,"survived"])
glimpse(raw_data)

set.seed(123)
train_test_split<-initial_split(raw_data, prop=0.65)

train_tbl<-training(train_test_split)
test_tbl<-testing(train_test_split)

glimpse(train_tbl)
#0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
train_tbl<-train_tbl[sample(nrow(train_tbl)),]
test_tbl<-test_tbl[sample(nrow(test_tbl)),]

#####################################
rec_obj <- recipe (survived ~., data = train_tbl) %>%
  step_knnimpute(all_predictors(), -all_outcomes())  %>%
  step_mutate(solo = (parch == 0) + (sibsp==0))  %>%
  step_mutate(crew = (sex=="male") + (fare==0))  %>%
  step_dummy(all_nominal() , -all_outcomes(), one_hot = TRUE) %>%
  prep(data = train_tbl)

rec_obj

train_ready<-bake(rec_obj, new_data=train_tbl)
test_ready<-bake(rec_obj, new_data=test_tbl)

glimpse(test_ready)

#Log Regression
Titanic.log <- glm (survived ~., data = train_ready, family="binomial")

Titanic.log.prob <- predict(Titanic.log, newdata=test_ready, type ="response")
Titanic.log.pred <- ifelse(Titanic.log.prob > 0.5, 1 , 0)

table(Titanic.log.pred, test_ready$survived)
mean(Titanic.log.pred==test_ready$survived)

###########KNN###########
train.X<-cbind(train_ready%>%select(everything()))
test.X<-cbind(test_ready%>%select(everything()))

train.label<-as.vector(train_ready$survived)
train.X<-train.X%>%select(-survived)
test.X<-test.X%>%select(-survived)
train.label
set.seed(1)

knn.pred<-knn(train.X,test.X,train.label,k=15)

table(knn.pred,test_ready$survived)
mean(knn.pred==test_ready$survived)
################################

##############RANDOM FOREST##############
###TO FIND THE OPTIMAL NUMBER OF TREES####
Titanic.rf <- randomForest(survived ~ ., data = train_ready)

plot(Titanic.rf)

importance(Titanic.rf)
varImpPlot(Titanic.rf)

Titanic.log.prob <- predict(Titanic.rf, newdata=test_ready, type ="response")
Titanic.log.prob

table(Titanic.log.prob, test_ready$survived)

############################################
glimpse(train_ready)
levels(train_ready$survived)<-c("No", "Yes")

control <- trainControl(method="repeatedcv",classProbs = TRUE, summaryFunction = twoClassSummary)

Titanic.rf.cv <- train(survived ~., method = "rf", data = train_ready, trControl = control,
                       metric = "ROC", ntree = 150, tuneLength = 2)  #he also tried 200 for ntree

plot(Titanic.rf.cv)
print(Titanic.rf.cv)

print(Titanic.rf.cv$finalModel)

Titanic.log.prob <- predict(Titanic.rf.cv, newdata=test_ready)
Titanic.log.prob

table(Titanic.log.prob, test_ready$survived)

###############################
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)

Titanic.gbm.cv <- train(survived ~., method = "gbm",data = train_ready, trControl = control, metric = "ROC", verbose = FALSE,tuneLength = 9)

Titanic.gbm.prob <- predict(Titanic.gbm.cv, newdata=test_ready)

Titanic.gbm.pred <- ifelse(Titanic.gbm.prob == "Yes", 1 , 0)

Titanic.gbm.pred
test_ready$survived

table(Titanic.gbm.pred,test_ready$survived)
table(Titanic.gbm.pred==test_ready$survived)
##############XGB MODEL################
y_train <- as.numeric(train_ready$survived)-1
y_train
y_test <- as.numeric(test_ready$survived)-1
y_test
#one hot encode of y.. need to make that for x too
trainm<-as.matrix(train_ready%>%select(-survived))
testm<-as.matrix(test_ready%>%select(-survived))

train_matrix<-xgb.DMatrix(data = trainm,label=y_train)
test_matrix<-xgb.DMatrix(data = testm, label=y_test)
glimpse(y_train)


Titanic.xgb <- xgboost(data = train_matrix, verbose = TRUE, eta = .005,
                       nrounds = 10000, prediction = TRUE, eval_metric ="error",
                       gamma=.1,subsample=1,
                       alpha = .9, lambda = .2, early_stopping = 20, 
                       objective = "binary:logistic", 
                       nfold = 150,                                                   # number of folds in K-fold
                       prediction = TRUE,                                           # return the prediction using the final model 
                       showsd = TRUE,                                               # standard deviation of loss across folds
                       stratified = TRUE,
                       early.stop.round = 100,
                       tree_method="exact"
                       )

glimpse(trainm)
imp<-xgb.importance(model = Titanic.xgb)
xgb.plot.importance(imp)
Titanic.xgb.prob <- predict(Titanic.xgb, test_matrix)
#model prediction
Titanic.xgb.pred <- ifelse(Titanic.xgb.prob > 0.5, 1,0)
#confusion matrix 
table(Titanic.xgb.pred,test_ready$survived)
mean(Titanic.xgb.pred ==test_ready$survived)

######GRID TUNING##############
modelLookup("xgbTree")

y_train <- train_ready$survived
y_train
y_test <- test_ready$survived
y_test



#y_train<-as.factor(pull(ifelse(y_train == 1, "Yes" , "No")))

#y_test<-as.factor(ifelse(y_test == 1, "Yes" , "No"))

y_test<-relevel(y_test,ref=2)
levels(y_test)
y_train<-relevel(y_train,ref=2)
levels(y_train)

y_train<-as.factor(ifelse(y_train == 1, "Yes" , "No"))

y_test<-as.factor(ifelse(y_test == 1, "Yes" , "No"))
y_test
make.names(y_test)


xgb_grid<-expand.grid(nrounds=2000,
                      max_depth=c(2,4,6,8,10),
                      eta=c(0.05,0.03,0.02,0.01,0.005,0.001),
                      gamma=c(0,1,2,3),
                      cosample_bytree=c(0,1,2,3),
                      min_child_weight=c(0,1,2,3),
                      subsample=c(0,1,2,3))

ctr<-trainControl(method = "repeatedcv",
                  number=5,
                  repeats = 2,
                  verboseIter = TRUE,
                  returnResamp = "all",
                  returnData=TRUE,
                  summaryFunction = twoClassSummary,
                  allowParallel = TRUE,
                  classProbs = TRUE)

BC_xgb <- train(trainm,
                y_train,
                trControl = ctr,
                tuneGrid = xgb_grid,
                method = "xgbTree",
                metric="ROC")

######################### hyper-parameter grid search
xgb_grid_1 = expand.grid(
  nrounds = 1000,
  # scale_pos_weight = 0.32, # uncommenting this line leads to the error
  eta = c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3),
  max_depth = c(2, 4, 6, 8),
  gamma = c(1, 2, 3), 
  subsample = c(0.5, 0.75, 1),
  min_child_weight = c(1, 2, 3), 
  colsample_bytree = 1
)

xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",
  classProbs = TRUE,
  allowParallel = TRUE
)

set.seed(1)

xgb_train_1 = train(
  x = as.matrix(Titanic_Full %>% select(-survived)),
  y = factor(training$survived, labels = c("Yes", "No")),
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree"
)



BC.xgbpred=predict(BC_xgb,newdata=xtest)
confusionMatrix(data = BC.xgbpred,ytest)

########LDA MODEL###########
#cross validated
library(MASS)
lda.fit<-lda(formula=survived~., data=train_ready)

summary(lda.fit)
names(lda.fit)
plot(lda.fit)
lda.pred <-predict(lda.fit,test_ready)
names(lda.pred)
lda.class<-lda.pred$class
##########CONFUSION MATRIX LDA####
table(lda.class, test_ready$survived)
##################################

#################!!!!!!!!!!DID NOT WORK!!!!!!!!!#################
##########QDA MODEL#############
qda.fit<-qda(formula=survived~., data=train_ready)

summary(qda.fit)
names(qda.fit)
plot(qda.fit)
qda.pred <-predict(qda.fit,test_ready)
names(qda.pred)
qda.class<-qda.pred$class
##########CONFUSION MATRIX QDA####
table(qda.class, test_ready$survived)
mean(qda.class== test_ready$survived)
##################################
#################!!!!!!!!!!DID NOT WORK!!!!!!!!!#################

