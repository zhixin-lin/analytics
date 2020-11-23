library(caret)
library(rpart)
library(rpart.plot) 
library(caTools)
library(dplyr)
library(randomForest)
library(ggplot2)
library(splines)
library(ModelMetrics)
setwd("D:/Academics/Harvard/2020 Fall/Analytics Edge/Assignments/HW3")

# Define a function for calculating out-of-sample R-squared
calc.OSR2 <- function(actuals, model.preds, baseline.preds) {
  return( 1 - sum((actuals - model.preds)^2) / sum((actuals - baseline.preds)^2) )
}

# Problem 1
boston <- read.csv("boston.csv")
set.seed(123)
split = createDataPartition(boston$nox, p = 0.7, list = FALSE)
boston.train = boston[split,]
boston.test = boston[-split,]

# 1.a) Linear regression model
mod.lm <- lm(nox~ dis, data=boston.train)
summary(mod.lm)

pred.lm <- predict(mod.lm, newdata = boston.test)
calc.OSR2(boston.test$nox, pred.lm, mean(boston.train$nox))
# R-squared is 0.5807, and out-of-sample R-squared is 0.6172861

# Plot
plot(nox ~ dis, data = boston.train)
abline(mod.lm)

# 1.b) Cubic Polynomial Regression
mod.poly = lm(nox ~ poly(dis, 3), data=boston.train)
summary(mod.poly)
pred.poly <- predict(mod.poly, newdata = boston.test)
calc.OSR2(boston.test$nox, pred.poly, mean(boston.train$nox))
# R-squared is 0.7111, and out-of-sample R-squared is 0.7224835

# Plot the cubic Polynomial Regression
ggplot(boston.train, aes(x = dis, y = nox, group = 1)) +
  geom_point(aes(col="blue")) + 
  geom_smooth(aes(col="red"), method = "lm", formula = "y ~ poly(x, 3)", se=FALSE)

# 1.c)
# Scatter Plots
ggplot(boston.train, aes(x = dis, group = 1)) +
  geom_point(aes(y = nox, colour = factor(nonretail)))
# New polynomial model with interaction terms
mod.poly2 = lm(nox ~ poly(dis, 3, raw = TRUE)*factor(nonretail), data = boston.train)
summary(mod.poly2)
pred.poly2 <- predict(mod.poly2, newdata = boston.test)
calc.OSR2(boston.test$nox, pred.poly2, mean(boston.train$nox))
# R-squared is 0.9056, out of sample R-squared is 0.9014088

# Plot
g <- ggplot(data = boston.train)
g <- g + geom_point(aes(x = dis, y = nox, colour = factor(nonretail))) 
g <- g + geom_smooth(aes(x = dis, y = nox, colour = factor(nonretail)),
                     method = "lm", formula = "y ~ poly(x, 3)", se=FALSE)
g

# 1.d) natural cubic spline model
knots <- quantile(boston.train$dis, p = c(0.2, 0.4, 0.6, 0.8))
mod.ns = lm(nox~ns(dis, knots = knots)*factor(nonretail),data=boston.train)
summary(mod.ns)
pred.ns <- predict(mod.ns, newdata = boston.test)
calc.OSR2(boston.test$nox, pred.ns, mean(boston.train$nox))
# R-squared is  0.9119, out of sample R-squared is 0.9045227
# The natural cubic spline model has the highest OSR2 among the four models, which
# indicates good prediction performance. Polynomial model also has very good OSR2 and
# it performs almost as well as the natural cubic spline model. Polynomial model without
# interaction terms perform only moderately well with an OSR2 of 0.72. The linear
# regression model has the lowest OSR2 of 0.617.

########################
########################


# Problem 2
# Read data, transform readmission column to fator, and split data sets
readmission = read.csv("readmission.csv")
readmission$readmission = as.factor(readmission$readmission)

set.seed(144)
split = createDataPartition(readmission$readmission, p = 0.75, list = FALSE)
readm.train <- readmission[split,]
readm.test <- readmission[-split,]

# 2.a) Data Exploration
ggplot(data = readm.train) +
  geom_bar(mapping = aes(x = age))
# Finding 1: the majority of the patients are between the age of 70 to 80.

ggplot(data = readm.train) +
  geom_bar(mapping = aes(x = age, fill = readmission), position="fill")
# Finding 2: the age group 20-30 has higher percentage of readmission patients than
# other age groups

# 2.b) Loss matrix Computation
cost.fp = 1200
cost.tn = 0
cost.fp - cost.tn
# the difference in costs between a false positive and a true negative is 1200

cost.fn = 35000
cost.tp = 1200 + 35000*0.75
cost.fn - cost.tp
# the difference in costs between a false negative and a true positive is 7550

# Define loss matrix
# TN FP 
# FN TP
LossMatrix = matrix(0,2,2)
LossMatrix[1,2] = cost.fp - cost.tn
LossMatrix[2,1] = cost.fn - cost.tp
LossMatrix

# 2.c) CART model
readm.mod = rpart(readmission~.,
                      data = readm.train,
                      parms=list(loss=LossMatrix),
                      cp=0.001)
prp(readm.mod)

# 2.d) CART model prediction
pred.cart = predict(readm.mod, newdata=readm.test, type="class")
# Confusion matrix
# TN FP 
# FN TP
confusion.matrix = table(readm.test$readmission, pred.cart)
confusion.matrix
# Accuracy
accuracy <- sum(diag(confusion.matrix))/sum(confusion.matrix)
accuracy
# True Positive Rate
TPR <- confusion.matrix[2,2]/sum(confusion.matrix[2,])
TPR
# False positive rate
FPR <- confusion.matrix[1,2]/sum(confusion.matrix[1,])
FPR
# TOtal cost of current practice with no telehealth intervention
cost.current = 35000 * sum(confusion.matrix[2,])
cost.current
# Total cost of applying the CART model
cost.mod = cost.fp * confusion.matrix[1,2] + cost.fn * confusion.matrix[2,1] +
  cost.tp * confusion.matrix[2,2]
cost.mod
# Difference in cost
cost.current - cost.mod
# Compared to no telehealth intervention, by applying the CART model we are able 
# to reduce total monetary cost from 99365000 to 96877750. The benefit of applying 
# the intervention based on the CART model is that it reduces overall monetary cost 
# by 2487250, but it is going to increase administrative efforts and man power in 
# order to implement this. And the flip side is true for the current practice 
# - no extra efforts but lose more money.

# 2.e) The total cost of current practice is 99365000 as calculated in d.
# To make the model cost-effective, cost.mod should be smaller than cost.current
# x represents the real cost of telehealth intervention
# Solving: x * 4565 + 35000 * 1784 + (x + 35000*0.75) * 1055< 99365000
# x < 1642.571
# Thus for cost lower than 1642.6 would the model return cost-effective recommendations

# 2.f) First of all, I would remove the loss matrix to avoid the model biasing towards
# one side. And then with minbucket fixed, I will do 10-fold cross validation to 
# find cp that minimizes the average loss over the folds. Furthermore, i could also
# run cross validation to tune the minbucket parameter. With the tuned parameters
# cp and minbucket found, we can then select the CART model for optimized prediction power.

########################
########################


# Problem 3

ames = read.csv("ames.csv")
set.seed(15071)
split = createDataPartition(ames$SalePrice, p = 0.7, list = FALSE)
ames.train = ames[split,]
ames.test = ames[-split,]

# 3.a) Linear regression model
mod1 <- lm(SalePrice~., data=ames.train)
summary(mod1)
# The coefficients of some variables are NA because they are not estimable due to
# the fact that they are linearly dependent on other variables or that there is not
# enough observations to estimate the relevant parameters.

# 3.b) CART cross validation
set.seed(144)
cv.trees = train(SalePrice~.,
                 data=ames.train,
                 method = "rpart",
                 trControl = trainControl(method = "cv", number = 10),
                 metric="RMSE", maximize=FALSE,
                 tuneGrid = data.frame(.cp = seq(.0002,.004,.0002)))
cv.trees
# The best cp that minimizes RMSE is 0.0004
# CART model
mod2 = rpart(SalePrice~.,
                  data = ames.train,
                  cp=0.0004)
prp(mod2)
# The independent variable and the cutoff of each split determines the prediction 
# direction of all data that falls into that same leaf. If we follow the path of 
# each split we will be able to arrive at the predicted price. The higher the 
# variable is in the tree hierarchy, the more primary role it plays in determining 
# the predicted price.


# 3.c) Random forest Cross Validation
set.seed(144)
rf.cv = train(y = ames.train$SalePrice,
              x = subset(ames.train, select=-c(SalePrice)),
              method="rf", nodesize=25, ntree=80,
              trControl=trainControl(method="cv", number=5),
              metric="RMSE",
              tuneGrid=data.frame(mtry=seq(1,10,1)))

rf.cv
# Select mtry = 9 and construct the random forest model
set.seed(144)
mod3 = randomForest(SalePrice~., data=ames.train, mtry=9, nodesize=25, ntree=80, importance = TRUE)
# Check for importance
importance(mod3, type=1)
important.variables <- importance(mod3)[order(importance(mod3)),]

# %IncMSE tells us how much worse the model would be if we lost the info contained in the variable
# The larger the value is, the more important the variable is.The top 3 most important variables are
# GrLivArea, LotArea, and X1stFlrSF.


# 3.d) Linear Regression in sample
pred1.train <- predict(mod1, newdata = ames.train)
calc.OSR2(ames.train$SalePrice, pred1.train, mean(ames.train$SalePrice))
mae(ames.train$SalePrice, pred1.train)
rmse(ames.train$SalePrice, pred1.train)
# Linear regression out of sample
pred1.test <- predict(mod1, newdata = ames.test)
calc.OSR2(ames.test$SalePrice, pred1.test, mean(ames.train$SalePrice))
mae(ames.test$SalePrice, pred1.test)
rmse(ames.test$SalePrice, pred1.test)
# Linear Regression summary: 
# In sample R-squared is 0.898, MAE is 14638.33, RMSE is 22476.65
# Out of sample R-squared is 0.8449963, MAE is 16123.29, RMSE is 27292.6

# CART in sample
pred2.train <- predict(mod2, newdata = ames.train)
calc.OSR2(ames.train$SalePrice, pred2.train, mean(ames.train$SalePrice))
mae(ames.train$SalePrice, pred2.train)
rmse(ames.train$SalePrice, pred2.train)
# CART out of sample
pred2.test <- predict(mod2, newdata = ames.test)
calc.OSR2(ames.test$SalePrice, pred2.test, mean(ames.train$SalePrice))
mae(ames.test$SalePrice, pred2.test)
rmse(ames.test$SalePrice, pred2.test)
# CART summary: 
# In sample R-squared is 0.9059837, MAE is 15255.79, RMSE is 21579.69
# Out of sample R-squared is 0.7814296, MAE is 22165.77, RMSE is 32409.29

# Random forest in sample
pred3.train <- predict(mod3, newdata = ames.train)
calc.OSR2(ames.train$SalePrice, pred3.train, mean(ames.train$SalePrice))
mae(ames.train$SalePrice, pred3.train)
rmse(ames.train$SalePrice, pred3.train)
# Random forest out of sample
pred3.test <- predict(mod3, newdata = ames.test)
calc.OSR2(ames.test$SalePrice, pred3.test, mean(ames.train$SalePrice))
mae(ames.test$SalePrice, pred3.test)
rmse(ames.test$SalePrice, pred3.test)
# Random forest summary: 
# In sample R-squared is 0.9375628, MAE is 11654.33, RMSE is 17585.93
# Out of sample R-squared is 0.876623, MAE is 16146.2, RMSE is 24349.55

# 3.e) This would not be a fair reflection. Because the parameters optimized based 
# on the test set might be overfitting to that specific data set but not generalize 
# to predict unseen data set well. So we might get better Rsqaured in the test set 
# but not so good in future data. We should avoid looking at the test set before 
# finish computing the trained model for fair validation.

# 3.f) I would recommend using random forest. Random forest captures non-linear 
# structures and has the best prediction performance based on OSR2, MAE, and RMSE, 
# but takes longest time to compute and is the most difficult one to interpret. 
# CART is easy to interpret, efficient to compute, but the prediction performance 
# is only moderately good. Linear regression is somewhat interpretable and fast 
# to compute, however the prediction performance is not so strong as it does not 
# capture non-linear structures. Overall, if I am a house buyer hoping to gain better 
# bargaining power, I am interested in predicting the price as accurately as possible 
# and does not care so much about interpretability and speed. And thus I will recommend
# random forest.