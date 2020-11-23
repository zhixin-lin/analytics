# Problem 1
setwd("D:/Academics/Harvard/2020 Fall/Analytics Edge/Assignments/HW1")
climate <- read.csv("climate_change.csv")

# Split data into training and testing sets
climateTrain <- climate[climate$Year <= 2002,]
climateTest <- climate[climate$Year > 2002,]

# Build Linear regression training model
mod1 <- lm(Temp~. -Year-Month, data=climateTrain)
summary(mod1)

# Calculate OSR2
pred <- predict(mod1, newdata = climateTest)
SSE <- sum((pred - climateTest$Temp)^2)
train.mean <- mean(climateTrain$Temp)
SST <- sum((train.mean - climateTest$Temp)^2)
1 - SSE/SST
# a) The training set R-squared is 0.6921. 
# The test set out-of-sample R-squared is 0.7771193

# b) MEI, CO2, N2O, CFC_11, CFC_12, TSI, Aerosols

# c) The third explanation:All of the gas concentration variables reflect human development 
# N2O and CFC-11 are correlated with other variables in the data set.

# Check for collinearity
cor(climateTrain)
# d) N2O is highly correlated with Co2, CH4, CFC_11, CFC_12
# e) CFC_11 is highly correlated with CO2, CH4, N2O, CFC_12

# Build a new model with reduced independent variables
mod2 <- lm(Temp~ N2O + TSI + Aerosols + MEI, data=climateTrain)
summary(mod2)

# Calcualte OSR2
pred2 <- predict(mod2, newdata = climateTest)
SSE2 <- sum((pred2 - climateTest$Temp)^2)
train.mean2 <- mean(climateTrain$Temp)
SST2 <- sum((train.mean2 - climateTest$Temp)^2)
1 - SSE2/SST2
# f) Multiple R-squared is 0.649.Out-of-sample R-squared is 0.8843635.
# Coefficients:
#  N2O          2.428e-02
#  TSI          8.577e-02
#  Aerosols    -1.725e+00
#  MEI          6.550e-02

# Problem 2
car <- read.csv("WranglerElantra2019.csv", fileEncoding = 'UTF-8-BOM')
carTrain <- car[car$Year <= 2018,]
carTest <- car[car$Year > 2018,]
Wrangler.mod1 <- lm(Wrangler.Sales~ Year + Unemployment.Rate + Wrangler.Queries + CPI.Energy + CPI.All, data=carTrain)
summary(Wrangler.mod1)

# a)i) Year, Wrangler.Queries are significant.

# a)ii)1) Year has a negative coefficient, which doesn't make sense and implies multicollinearity. 
# I then check for correlations
cor(carTrain[c(4,5,7,8,10,11)])

# Wrangler.Queries is highly correlated with Year, Unemployment.Rate and CPI.All
# I thus drop Year, Unemployment.Rate, and CPI.All,
# and computes a new model with only Wrangler.Queries and CPI.Energy
Wrangler.mod2 <- lm(Wrangler.Sales~ Wrangler.Queries + CPI.Energy, data=carTrain)
summary(Wrangler.mod2)
# both variables are significant with a R-squared of 0.726

# a)ii)2) The new model equation is:
# Wrangler.Sales = -8069.46 + 226.64*(Wrangler.Queries) + 35.48*(CPI.Energy)
# An additional number of Wrangler query is expected to be associated with an 
# increase of 226.64 number of Wrangler units sold in a given month and year
# An additional unit of CPI.Energy is expected to be associated with an increase
# of 35.48 number of Wrangler units sold in a given month and year

# a)ii)3) They make sense. The number of Google search queries shows people's
# interests in buying Wrangler, and thus the coefficient should be positive.
# For CPI of energy, higher energy price means higher demands for fuel exist, 
# which is related to higher demands of car purchases. Thus positive sign.

# a)iii) 
pred.Wrangler <- predict(Wrangler.mod2, newdata = carTest)
SSE.Wrangler <- sum((pred.Wrangler - carTest$Wrangler.Sales)^2)
train.Wranglermean <- mean(carTrain$Wrangler.Sales)
SST.Wrangler <- sum((train.Wranglermean - carTest$Wrangler.Sales)^2)
1 - SSE.Wrangler/SST.Wrangler
# R-squared of the new trained model is 0.726, which predict training-set 
# observations moderately well.
# OSR-squared is 0.8186823 and higher than R-squared,indicative of good 
# out of sample predictive performance.

# b)i) Wrangler queries increased for the firs half of the year and dropped in 
# the second half. As weather gets warmer, people are more motivated to plan for outdoor
# sports activities, and have a higher interests in buying Wrangeler. Vice Versa.

# b)ii) 
Wrangler.mod3 <- lm(Wrangler.Sales~ Wrangler.Queries + CPI.Energy + Month.Factor, data=carTrain)
summary(Wrangler.mod3)
# The new model takes into consideration of categorical variables of months.
# The baseline reference month is April, and the coefficient of 11 Month.Factor variables means
# the increase/decrease of sales numbers compared to April.
# Month.FactorMay: Compared to April, number of Wrangle units sold in May increases by 502.27
# Month.FactorJune: Compared to April, number of Wrangle units sold in June decreases by 1108.13
# Month.FactorJuly: Compared to April, number of Wrangle units sold in July decreases by 1916.39
# Month.FactorAugust: Compared to April, number of Wrangle units sold in Aug decreases by 1846.64
# Month.FactorSeptember: Compared to April, number of Wrangle units sold in Sep decreases by 2461.65
# Month.FactorOctober: Compared to April, number of Wrangle units sold in Oct decreases by 2392.30
# Month.FactorNovember: Compared to April, number of Wrangle units sold in Nov decreases by 2954.25
# Month.FactorDecember: Compared to April, number of Wrangle units sold in Dec decreases by 1047.69
# Month.FactorJanuary: Compared to April, number of Wrangle units sold in Jan decreases by 5065.49
# Month.FactorFebruary: Compared to April, number of Wrangle units sold in Feb decreases by 3980.98
# Month.FactorMarch: Compared to April, number of Wrangle units sold in March decreases by 465.05

# b)iii)Wrangler.Queries, CPI.Energy, Month.FactorAugust, Month.FactorFebruary, 
# Month.FactorJanuary, Month.FactorJuly, Month.FactorNovember, Month.FactorOctober,
# Month.FactorSeptember are significant.
pred.Wrangler2 <- predict(Wrangler.mod3, newdata = carTest)
SSE.Wrangler2 <- sum((pred.Wrangler2 - carTest$Wrangler.Sales)^2)
train.Wranglermean2 <- mean(carTrain$Wrangler.Sales)
SST.Wrangler2 <- sum((train.Wranglermean2 - carTest$Wrangler.Sales)^2)
1 - SSE.Wrangler2/SST.Wrangler2
# The training set R-squared is 0.8385. The test set is OSR-squared is 0.9166378

# b)iv)The model has been improved by add Month.Factor because adjusted-R has
# increased compared to the previous model and OSR-squared stay close to R-squared.

# b)v) I could also use Month.Numeric as an independent variable to model seasonality.
# I don't think this will improve the model compared to using Month.Factor,
# because the relationship between the numeric value of month and Wrangler
# queries is not linear.

# c)i) I start by building a model with all 5 variables
Elantra.mod1 <- lm(Elantra.Sales~ Year + Unemployment.Rate + Elantra.Queries + CPI.Energy + CPI.All, data=carTrain)
summary(Elantra.mod1)
# Unemployment.Rate, Elantra.Queries, CPI.Energy, and CPI,All are significant.
# I then built a new model with theses significant factors

Elantra.mod2 <- lm(Elantra.Sales~ Unemployment.Rate + Elantra.Queries + CPI.Energy + CPI.All, data=carTrain)
summary(Elantra.mod2)

pred.Elantra <- predict(Elantra.mod2, newdata = carTest)
SSE.Elantra <- sum((pred.Elantra - carTest$Elantra.Sales)^2)
train.Elantramean <- mean(carTrain$Elantra.Sales)
SST.Elantra <- sum((train.Elantramean - carTest$Elantra.Sales)^2)
1 - SSE.Elantra/SST.Elantra
# Training set R-squared is 0.3403, test set OSR-squred is 0.320462

# c)ii)
date.Train <- as.Date(carTrain$date, format = "%m/%d/%Y")
library(ggplot2)
ggplot(carTrain, aes(date.Train, group = 1)) +
  geom_line(aes(y = Wrangler.Sales, colour = "Wrangler.Sales")) +
  geom_line(aes(y = Elantra.Sales, colour = "Elantra.Sales"))
# Seasonality could also be observed in the sales of Elantra, and thus adding
# seasonality will improve the prediction power of the model

# d)i) Producing more: warehouse/storage fees, production variable cost, worker salary
# Producing fewer: loss of revenues

# d)ii) I think the cost of overproduction is larger than the costs of underproduction
# Since automobile industry is asset-heavy, overproduction will greatly increase
# inventory management and storage cost. In the long run, it will also decrease market.
# demands and price.Whereas the major cost of underproduction is related to loss of revenues. 
# Automobile purchase is usually non-urgent and customers have more patience to wait
# for restock. This may even be turned into a hunger-marketing strategy.
# Thus I will recommend producing fewer than the predicted demands.

# d)iii) We may consider introducing a categorical variable isHigher in the linear
# regression model. When overproduction is more costly, isHigher is true, whereas
# the opposite sets it to false. In the training data set, when isHigher is true, 
# demands are penalized by a certain percentage based on the difference in cost.
# We could then run the model based on the adjusted data, which will account for
# the differences in overproduct/underproduction cost.