library(lars)
library(glmnet)
library(plyr)
library(ggplot2)
library(ggcorrplot)
library(caret)
library(leaps)
library(xgboost)
library(cluster)

setwd("D:/Academics/Harvard/2020 Fall/Analytics Edge/Assignments/HW4")
Hitters_raw <- read.csv("Hitters.csv")

# Define a function for calculating out-of-sample R-squared
calc.OSR2 <- function(actuals, model.preds, baseline.preds) {
  return( 1 - sum((actuals - model.preds)^2) / sum((actuals - baseline.preds)^2) )
}


# Problem 1: Moneyball Analytics
# 1.a)i) correlation matrix
cor(Hitters_raw[,2:18])
ggcorrplot(cor(Hitters_raw[,2:18]))
# Observation: offensive statistics(CAtBat, CHits, CHmRun, CRuns, CRBI, CWalks) over 
# the player's career are highly positively correlated with each other. Offensive 
# statistics during the season are moderately positively correlated with each other. 
# In defensive statistics, assists and errors are positively correlated.

# 1.a)ii) CRBI and CRuns have higher correlation with salary.
# Plot salary as a function of CRBI
ggplot(data=Hitters_raw, aes(x=CRBI, y=Salary)) + 
  geom_point() +
  geom_smooth(method='lm')
# Increase in CRBI is connected to an increase in Salary

# Plot salary as a function of CRuns
ggplot(data=Hitters_raw, aes(x=CRuns, y=Salary)) + 
  geom_point() +
  geom_smooth(method='lm')
# Increase in CRuns is also connected to an increase in Salary

# Feature normalization and splitting data into training and test sets
pp <- preProcess(Hitters_raw, method=c("center", "scale"))
Hitters <- predict(pp, Hitters_raw)
set.seed(15071)
train.obs <- sort(sample(seq_len(nrow(Hitters)), 0.7*nrow(Hitters)))
train <- Hitters[train.obs,2:21]
test <- Hitters[-train.obs,2:21]

# 1.b)i) Linear regression model
mod.lm = lm(Salary ~ ., data=train)
summary(mod.lm)
pred.lm <- predict(mod.lm, newdata = test)
calc.OSR2(test$Salary, pred.lm, mean(train$Salary))
# R-squared is 0.5603
# OSR-squared is 0.4003593
# Comments: OSR-squared shows that the prediction power of the model is not good.
# R-squared is much higher than OSR-squared, indicating model overfitting.
# AtBat(negative), Hits(positive), Walks(positive), CWalks(negative), PutOuts(positive), 
# and DivisionW(negative) are significant variables.
# This result is not entirely in line with the observations. For example, neither 
# CRBI or CRuns were significant variables in the model. The linear
# regression model shows that signs of AtBat and CWalks are negative. But in the 
# correlation matrix they were positively correlated with salary. 

# 1.b)ii) Restricted linear regression model
mod.lm2 = lm(Salary ~ AtBat+Hits+Walks+CWalks+PutOuts+Division, data=train)
summary(mod.lm2)
pred.lm2 <- predict(mod.lm2, newdata = test)
calc.OSR2(test$Salary, pred.lm2, mean(train$Salary))
# R-squared is 0.4452
# OSR-squared is 0.4505801

# 1.c)i) Ridge regression model
all.lambdas <- c(exp(seq(15, -10, -.1)))
hitters.grid.l2 <- expand.grid(alpha = 0,lambda=all.lambdas)
set.seed(1)
hitters.cv.l2 <- train(y = train$Salary,
                        x = data.matrix(subset(train, select=-c(Salary))),
                        method = "glmnet",
                        trControl = trainControl(method="cv", number=10),
                        tuneGrid = hitters.grid.l2)
hitters.cv.l2$bestTune
# Best lambda of the ridge regression model is 0.04978707
# Plot the cross-validated Mean Squared Error as a function of lambda.
ggplot(data = hitters.cv.l2$results, aes(x = lambda, y = RMSE^2)) + 
  geom_point() + 
  scale_x_log10()

# LASSO model
hitters.grid.l1 <- expand.grid(alpha = 1, lambda=all.lambdas)
set.seed(1)
hitters.cv.l1 <- train(y = train$Salary,
                       x = data.matrix(subset(train, select=-c(Salary))),
                       method = "glmnet",
                       trControl = trainControl(method="cv", number=10),
                       tuneGrid = hitters.grid.l1)
hitters.cv.l1$bestTune
# Best lambda of the LASSO model is 0.004991594
# Plot the cross-validated Mean Squared Error as a function of lambda.
ggplot(data = hitters.cv.l1$results, aes(x = lambda, y = RMSE^2)) + 
  geom_point() + 
  scale_x_log10()

# 1.c)ii) 
# Re-train the ridge regression model on the full training set
lambda.l2 = 0.04978707
hitters.l2 = glmnet(x = data.matrix(subset(train, select=-c(Salary))), 
                       y = train$Salary, 
                       alpha = 0, 
                       lambda = lambda.l2)

# Re-train the LASSO model on the full training set
lambda.l1 = 0.004991594
hitters.l1 = glmnet(x = data.matrix(subset(train, select=-c(Salary))), 
                    y = train$Salary, 
                    alpha = 1, 
                    lambda = lambda.l1)

# Reporting Coefficients
beta.ridge <- hitters.l2$beta
beta.ridge
beta.lasso <- hitters.l1$beta
beta.lasso
sum(beta.ridge != 0)
sum(beta.lasso != 0)
# LASSO has 15 non-zero coefficients whereas ridge regression has 19. Ridge regression's
# coefficients on average are smaller than that of LASSO.


# Reporting R-squared
# In sample R-squared of the ridge regression model is 0.5286452
hitters.l2$dev.ratio
# out of sample R-squared of the ridge regression model is 0.474942
hitters.preds.l2 <- predict(hitters.l2, 
                            newx = data.matrix(subset(test, select=-c(Salary))), 
                            s=lambda.l2)
calc.OSR2(test$Salary, hitters.preds.l2, mean(train$Salary))
# In sample R-squared of the LASSO model is 0.5557009
hitters.l1$dev.ratio
# out of sample R-squared of the LASSO model is 0.4321668
hitters.preds.l1 <- predict(hitters.l1, 
                            newx = data.matrix(subset(test, select=-c(Salary))), 
                            s=lambda.l1)
calc.OSR2(test$Salary, hitters.preds.l1, mean(train$Salary))
# Comments on R-squared: LASSO's in-sample R-squared is higher than ridge regression.
# But in terms of out-of-sample-performance, ridge regression is better than LASSO.


# 1.c)iii) Yes. LASSO has 15 non-zero coefficients whereas ridge regression has 19.

# 1.d)i) I got 6 predictors in question b)ii)
set.seed(1)
lasso.cv <- cv.glmnet(y = train$Salary,
          x = data.matrix(subset(train, select=-c(Salary))),
          lambda = all.lambdas)

all_constrained_validation_runs <- lasso.cv$nzero <= 6
best_constrained_run <- which.min(lasso.cv$cvm[all_constrained_validation_runs])
best_constrained_lambda = lasso.cv$lambda[best_constrained_run]
best_constrained_lambda
# Best constrained lambda is 0.0450492
lasso = glmnet(x = data.matrix(subset(train, select=-c(Salary))), 
                    y = train$Salary, 
                    alpha = 1, 
                    lambda = best_constrained_lambda)
# In sample R-squared of the constrained LASSO model is 0.4576504
lasso$dev.ratio
# out of sample R-squared of the constrained LASSO model is 0.5166173
lasso.preds <- predict(lasso, newx = data.matrix(subset(test, select=-c(Salary))), 
                       s=best_constrained_lambda)
calc.OSR2(test$Salary, lasso.preds, mean(train$Salary))
# The constrained LASSO model has higher OSR2 than the previous LASSO model and the
# linear regression model

# Predictors
beta.lasso.new <- lasso$beta
beta.lasso.new
# 6 Predictors chosen by the models are Hits, Walks, CHits, CRBI, PutOuts, and Division
# 6 Predictors chosen by me were AtBat, Hits, Walks, CWalks, PutOuts, and Division
# 4 predictors were the same and 2 were different

# 1.d)ii)
set.seed(1)
fs <- train(Salary~., train,
            method = "leapForward",
            trControl = trainControl(method = "cv", number = 10),
            tuneGrid = expand.grid (.nvmax=seq(1,15)))
summary(fs)
# The size of the best subset is 10. Variables includes AtBat, Hits, Walks, Years,
# CRuns, CRBI, CWalks, PutOuts, Assists, and Division. This subset is larger than my
# selection in b)ii). 6 out of the 6 predictors that I have manually chosen were also
# chosen in the leap forward method. Whereas only 4 out of 6 predictors chosen by the 
# constrained LASSO model were also chosen by the leap forward method

# Build a linear regression model using the selected 10 variables
fs.mod <- lm(Salary~AtBat+Hits+Walks+Years+CRuns+CRBI+CWalks+PutOuts+Assists+Division, data = train)
summary(fs.mod)
forward.pred <- predict(fs.mod, newdata = test)
calc.OSR2(test$Salary, forward.pred, mean(train$Salary))
# in-sample-R-squared is 0.552
# OSR2 of the forward stepwise model is 0.4489453


# 1.e) XGBoost
set.seed(1)
xgb.cv <- train(y = train$Salary,
                         x = data.matrix(subset(train, select=-c(Salary))),
                         method = "xgbTree", verbosity=0,
                         trControl = trainControl(method="cv", number=5))
# Reporting hyperparameters
xgb.cv$bestTune
# nrounds:50 max_depth:3 eta:0.3 gamma:0 colsample_bytree:0.8  
# min_child_weight:1 subsample:1

set.seed(1)
xgb <- xgboost(data = data.matrix(subset(train, select=-c(Salary))), 
               label = train$Salary, nrounds = 50, max_depth = 3, eta = 0.3, 
               gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, subsample = 1, 
               verbose = 0)
# in-sample-R-squared of XGBoost is 0.9934909
xgb.preds.train <- predict(xgb, newdata = data.matrix(subset(train, select=-c(Salary))))
calc.OSR2(train$Salary, xgb.preds.train, mean(train$Salary))
# OSR2 of the XGBoost is 0.5955549
xgb.preds <- predict(xgb, newdata = data.matrix(subset(test, select=-c(Salary))))
calc.OSR2(test$Salary, xgb.preds, mean(train$Salary))
# OSR2 of the XGBoost model is the highest among all models calculated so far


########################
########################


# Problem 2
data = read.csv("returns.csv")
returns = data[,3:122]
# 2.a) Reporting the number of companies in each industry sector
count(data, "Industry")
# average stock return between 01/2008 and 12/2010 for each sector
agg<-aggregate(returns[,23:58],by=list(data$Industry),FUN=mean)
#  Transpose form for easier plotting
agg.transpose = data.frame(t(agg[-1]))
colnames(agg.transpose) <- agg[, 1]
agg.transpose <- cbind(rownames(agg.transpose), data.frame(agg.transpose, row.names=NULL))
names(agg.transpose)[1] <- "Month"
# Clean up the month column
data.months <- substring(agg.transpose$Month, 4)
data.months <- lubridate::parse_date_time(data.months, "Ym")
# Plot the average stock return between 01/2008 and 12/2010 for each sector
ggplot(agg.transpose,aes(x=data.months, group=1)) + 
  geom_line(aes(y=Consumer.Discretionary,color="Consumer.Discretionary")) + 
  geom_line(aes(y=Consumer.Staples,color="Consumer.Staples")) +
  geom_line(aes(y=Energy,color="Energy")) + 
  geom_line(aes(y=Financials,color="Financials")) + 
  geom_line(aes(y=Health.Care,color="Health.Care")) + 
  geom_line(aes(y=Industrials,color="Industrials")) + 
  geom_line(aes(y=Information.Technology,color="Information.Technology")) + 
  geom_line(aes(y=Materials,color="Materials")) + 
  geom_line(aes(y=Telecommunications.Services,color="Telecommunications.Services")) + 
  geom_line(aes(y=Utilities,color="Utilities")) +
  labs(x="months", y="return")

# The material industry had the worst average return across the stocks in September 2008
agg$Group.1[(which.min(agg$avg200809))]
# industry information is NOT sufficient for investors to build a portfolio 
# diversification strategy because there is great return variability of companies
# within the same industry.

# 2.b) We don't need to do normalization because all the columns are essentially 
# of the same metric type(return), which has the same unit and scale.

# Hierarchical clustering
d <- dist(returns)
hclust.mod <- hclust(d, method="ward.D2")

# Plot dendrogram
plot(hclust.mod, labels=F, ylab="Dissimilarity", xlab = "", sub = "")

# the scree Plot
hc.dissim <- data.frame(k = seq_along(hclust.mod$height),
                        dissimilarity = rev(hclust.mod$height))
head(hc.dissim)
plot(hc.dissim$k, hc.dissim$dissimilarity, type="l", 
     ylab="Dissimilarity", xlab = "Number of Clusters")
# Looks like the knee in the curve is somewhere between 0 and 60
# Zooming in the scree plot
plot(hc.dissim$k, hc.dissim$dissimilarity, type="l", xlim=c(0,60),
     ylab="Dissimilarity", xlab = "Number of Clusters")
# Further zooming in between 0 and 20
plot(hc.dissim$k, hc.dissim$dissimilarity, type="l", xlim=c(0,20),
     ylab="Dissimilarity", xlab = "Number of Clusters")
axis(side = 1, at = 1:15)
# 8 is a reasonable number of clusters because dissimilarity ceases to drop significantly
# beyond the point. The curves gradually flattens out beyond number 8.


# 2.c) construct the clusters with number 8
h.clusters <- cutree(hclust.mod, 8)
# Number of companies in each cluster
table(h.clusters)
# Number of companies per industry sector in each cluster
table(data$Industry, h.clusters)
# the average returns by cluster in October 2008 and in March 2009
agg2 <- aggregate(returns[c("avg200810", "avg200903")], by=list(h.clusters), mean)
agg2
summary(agg2)
# Characterize each cluster qualitatively:
# 1: hit relatively light by the recession and rebound relatively slow; 
#    diverse industries; the largest cluster
# 2: hit moderately hard by the recession and rebound relatively slow; 
#    majority in Financials and Consumer Discretionary; 
#    Some in Industries and IT sectors; Second largest cluster
# 3: low volatility overall;majority in Utilities, some in Healthcare and Consumer Staples
# 4: hit moderately hard by the recession and rebound slowly; Majority in Energy sector
# 5: hit hard by the recession and rebound very quickly; one special case in the Financial sector
# 6: hit lightly by the recession and rebound moderately; Majority in Financial sector
# 7: hit relatively hard and rebound moderately; Majority in Consumer Discretionary sector
# 8: hit hard by recession and rebound slowly;financial sector concentrated;very small cluster

# 2.d) K-means clustering
set.seed(1)
km <- kmeans(returns, centers = 8, iter.max=100) 
# Cluster centroids in October 2008 and in March 2009
agg.kmeans <- aggregate(returns[c("avg200810", "avg200903")], by=list(km$cluster), mean)
agg.kmeans
# Comparing Number of companies in each cluster in K-means and Hierarchical clustering
table(km$cluster)
table(h.clusters, km$cluster)

# Comparing Number of companies per industry sector in each cluster
table(data$Industry, km$cluster)
table(data$Industry, h.clusters)
# There are 4 cluster well-matches between the two models
# K means' cluster 3 matches with HC's 7, 5 with 5, 7 with 4, and 8 with 1
# Besides these, K means's cluster 1 is similar to HC's cluster 6, but also includes
# a number of observations of HC's cluster 2.
# K mean's cluster 4 is consited of most of the companies of the consumer discretionary 
# and IT subset of HC's cluster 2.
# K mean's cluster 6 combines the majority of of HC's cluster 7 and some of cluster 1.
# K means' cluster 2 looks very different from each of the cluster in HC.

# 2.e) Hierarchical clusters silhouette score
sil.hc <- silhouette(h.clusters, d)
summary(sil.hc)
# The overall mean of hierarchical clusters is -0.01105
# the mean scores of each cluster are -0.08203118, -0.07783739, 0.27088123, -0.02796159, 
# 0.00000000, -0.01092169, 0.08093131  0.06602487

# K-means silhouette score
sil.kms <- silhouette(km$cluster, d)
summary(sil.kms)
# The overall mean of K-means clusters is 0.020862
# the mean scores of each cluster are: 0.03510772, -0.05650709, 0.01659775, -0.07156660, 
# -0.14095886, 0.17314589, 0.04436250 -0.03560910 

# Plotting the results
plot(h.clusters, col=1:8, border=NA)
plot(km$cluster, col=1:8, border=NA)

# 2.f) My model categorize stocks according to their monthly returns pattern. There
# are 8 different clusters showing different return and variability pattern.
# For example, some cluster exhibits strong volatility, whereas other cluster is very stable.
# To help diversify the stock portfolio, investors are advised to choose stocks from different 
# clusters instead of concentrating on one single cluster. Stocks within the same 
# cluster has strong correlation. Investor shall avoid putting eggs in the same basket. 
# This way investors would likely be able to have more stable long-term returns.
