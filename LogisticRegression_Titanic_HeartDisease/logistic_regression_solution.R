library(ggplot2)
library(caret)
library(ROCR)

# Problem 1
setwd("file_path")
titanic <- read.csv("titanic.csv")
titanic$Pclass <- factor(titanic$Pclass)
titanic$Sex <- factor(titanic$Sex)

# a)
mod1 = glm(Survived ~ Pclass + Sex + SibSp, data=titanic, family="binomial")
summary(mod1)

# b) Compared to first class, second class had lower probability of survival. 
# Third class passengers had even lower probability than that of second class.
# Compared to female, male had lower probability of survival
# Higher number of siblings and spouses would result in lower probability of survival

# c) 
exp(-0.24651*2)
# all else being equal, having two additional siblings/spouses aboard affect
# the odds of a passenger surviving by a factor of 0.6107791

# d) 
exp(-0.84708)/exp(-1.86648)
# the odds of survival becomes 2.77 times higher going from the third class to the
# second class

# e) it is not possible to have a numeric answer as the change in probability will depend
# on other independent variables.
# P_Class2(Y = 1) / P_Class3(Y = 1) 
# = (1/(1 + e^-(beta0 + betaclass2 + beta3 * sexmale + beta4 * SibSp)) / 
# (1/(1 + e^-(beta0 + betaclass3 + beta3 * sexmale + beta4 * SibSp))
# = (1 / (1 + 1/(OddsClass1 * betaClass2))) / (1 / (1 + 1/(OddsClass1 * betaClass3)))
# = (e^betaClass2 + OddsClass1 * e^(betaClass2 + betaClass3))/
# (e^betaClass3 + OddsClass1 * e^(betaClass2 + betaClass3))
# From the function above, the base case OddsClass1 could not be canceled out,
# and thus we need to know the value of Sibsp and sex to determine the change in survival
# probablity.


# Problem 2
# a) costMed = 172500 * p / 2.3 + 7500 * (1 - p / 2.3)
# costNoMed = 165000 * p 
# Solving: 172500 * p / 2.3 + 7500 * (1 - p / 2.3) < 165000 * p
# p > 0.08041
# I would recommend the medication when p is greater than 8.04%

library(caTools)
data = read.csv("framingham.csv")
data$TenYearCHD <- factor(data$TenYearCHD)
data$male <- factor(data$male)
data$currentSmoker <- factor(data$currentSmoker)
data$BPMeds <- factor(data$BPMeds)
data$prevalentStroke <- factor(data$prevalentStroke)
data$prevalentHyp <- factor(data$prevalentHyp)
data$diabetes <- factor(data$diabetes)
set.seed(31)
N <- nrow(data)
idx = sample.split(data$TenYearCHD, 0.75)
train <- data[idx,]
test = data[!idx,]

# b) 
mod2 = glm(TenYearCHD ~., data=train, family="binomial")
summary(mod2)
# The important factors include male, age, cigsPerDay, sysBP, and glucose
# These factors all impact TenYearCHD positively. they do make intuitive sense.
# Male is likely to be prone to risk-increasing behaviors such as smoking and drinking. 
# As people get older, their overall health conditions tend to get worse. 
# Cigarette is considered bad for health. High blood pressure and blood glucose 
# level are direct indicators of unhealthy condition.

# c)
new_patient <- data.frame()
new_patient[1,'male'] = '1'
new_patient[1,'age'] = 55
new_patient[1,'education'] = 'College'
new_patient[1,'currentSmoker'] = '1'
new_patient[1,'cigsPerDay'] = 10
new_patient[1,'BPMeds'] = '0'
new_patient[1,'prevalentStroke'] = '0'
new_patient[1,'prevalentHyp'] = '1'
new_patient[1,'diabetes'] = '0'
new_patient[1,'totChol'] = 220
new_patient[1,'sysBP'] = 140
new_patient[1,'diaBP'] = 100
new_patient[1,'BMI'] = 30
new_patient[1,'heartRate'] = 60
new_patient[1,'glucose'] = 80

pred <- predict(mod2, newdata=new_patient, type="response")
head(pred)
# the predicted probability is 0.2662407 , which is greater than threshold 0.08041
# the physician should prescribe the med

# d)
# Talking point 1: quit smoking
new_patient[1,'currentSmoker'] = '0'
new_patient[1,'cigsPerDay'] = 0
pred <- predict(mod2, newdata=new_patient, type="response")
head(pred)
# If the patient quits smoking, the probability that the he will experience CHD within 
# the next 10 years will decrease from 0.2662 to 0.2170

# Talking point 2: reduce blood pressure by exercising more, reducing sodium intake, etc.
new_patient[1,'sysBP'] = 120
new_patient[1,'diaBP'] = 80
pred <- predict(mod2, newdata=new_patient, type="response")
head(pred)
# The patient's blood pressure is higher than the normal range.If the patient reduces
# his sysBP to 120 and diaBP to 80, the probability that the he will 
# experience CHD within the next 10 years will further decrease to 0.1860

# Overall, if the patient could both quit smoking and reduce blood pressure, his probability 
# of getting CHD within the next 10 years will reduce from 0.2662 to 0.1860

# e) lifestyle and health changes might lead to improvement in multiple variables,
# which creates the issue of multicollinearity. For example, by losing weight one
# could improve BMI, blood pressure, total cholesterol, etc.

# f)
predTest = predict(mod2, newdata=test, type="response")
threshPred = (predTest > 0.08041)

#confusion matrix
confusion.matrix = table(test$TenYearCHD, threshPred)
confusion.matrix

# accuracy
accuracy = sum(diag(confusion.matrix)) / sum(confusion.matrix)
accuracy

# True positive rate
truePos = confusion.matrix[2,2]/sum(confusion.matrix[2,])
truePos

# False positive rate
falsePos = confusion.matrix[1,2]/sum(confusion.matrix[1,])
falsePos

# expected economic cost for patients in test set
cost <- confusion.matrix[1,2]*7500+confusion.matrix[2,1]*165000+
  confusion.matrix[2,2]*(172500*(1/2.3)+7500*(1-1/2.3))
cost
# Accurracy is  0.4354. True positive rate is 0.8849. False positive rate is 0.6452
# the expected economic cost is 16136413
                                                                                       
# g) baseline model
costBase <- 165000 * sum(confusion.matrix[2,])
costBase

# perfect model
costPft <- sum(confusion.matrix[2,]) * (172500 *(1 / 2.3) + 7500 * (1 - 1/2.3))
costPft

# If no one receives medication, the economic cost for patients in the test set
# is 22935000. If only patients who would otherwise get CHD are give med, the economic
# cost is 14299565. For our trained model, we set the bar for prescribing med to 
# 8.04% probability of developing CHD based on a cost-benefit analysis. This low 
# threshold resulted in a relatively low accuracy, but we prioritize the case that 
# people who would otherwise develop the disease do receive med. The economic cost 
# associated with the trained model is 16136413, which is slightly higher than the 
# ideal situation but still significantly less than the no-med situation. Hence the 
# model is effective in helping to reduce overall economic cost.

# h)
rocr.pred = prediction(predTest, test$TenYearCHD)
plot(performance(rocr.pred, "tpr", "fpr"))
abline(0,1)

AUC = as.numeric(performance(rocr.pred, "auc")@y.values)
AUC
# AUC is 0.7350

# i) I have chosen sysBP, age, and male based on relative significance.
mod3 <- glm(TenYearCHD~ sysBP + age + male, data = train, family="binomial")
summary(mod3)
predTest2 = predict(mod3, newdata=test, type="response")
threshPred2 = (predTest2 > 0.08041)

#confusion matrix
confusion.matrix2 = table(test$TenYearCHD, threshPred2)
confusion.matrix2

# accuracy
accuracy2 = sum(diag(confusion.matrix2)) / sum(confusion.matrix2)
accuracy2

# True positive rate
truePos2 = confusion.matrix2[2,2]/sum(confusion.matrix2[2,])
truePos2

# False positive rate
falsePos2 = confusion.matrix2[1,2]/sum(confusion.matrix2[1,])
falsePos2

rocr.pred2 = prediction(predTest2, test$TenYearCHD)
AUC2 = as.numeric(performance(rocr.pred2, "auc")@y.values)
AUC2
# The AUC of the new trained model is 0.7229, which performs moderately well differentiating
# who is likely to develop CHD and who is not.

# j) The factor gender raises ethical concern. It might be the behavioral tendency of 
# different gender rather than gender itself that caused the statistical difference in
# the likelihood of developing CHD. By using sex as a factor we are imposing
# stereotype on people. I would remove the male factor, run the model again, and chooses
# important factors to construct a simplified model.
mod4 <- glm(TenYearCHD~.-male, data = train, family="binomial")
summary(mod4)

# age, cigsPerDay, sysBP, and glucose now all have very high significance. We could
# then choose out of these factors to reconstruct a simplified model. For example,
mod5 <- glm(TenYearCHD~age + currentSmoker + sysBP, data = train, family="binomial")
summary(mod5)
