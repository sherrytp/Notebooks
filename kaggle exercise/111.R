
### Clean data for modeling

dfcleaning <- function(df){
  # trim outliers - majority close to 99 percentile
  df$avhv[df$avhv>=450] <- 450
  df$incm[df$incm>=130] <- 130
  df$inca[df$inca>=150] <- 150
  df$plow[df$plow>=56] <- 56
  df$npro[df$npro>=132] <- 132
  df$tgif[df$tgif>=402] <- 402
  df$lgif[df$tgif>=156] <- 156
  df$rgif[df$rgif>=60] <- 60
  df$agif[df$rgif>=35] <- 35
  # new attributes from decision tree
  # child
  df$chld_0 <- ifelse(df$chld ==0, 1,0)
  df$chld_ThM <- ifelse(df$chld>=3, 1,0)
  # hinc --middle 4 best, two side diff behivor
  df$hinc_4 <- ifelse(df$hinc==4, 1,0) # for donor amt, 1-4, 5-7
  # wrat
  df$wrat_0_4 <- ifelse(df$wrat<=4,1,0)
  # New attributes base on business knoewledge
  df$GiftTimes <- df$tgif/df$agif # total gifts divided by avg should be the total reposed times
  df$ResponseRate <- df$GiftTimes/df$npro
  # Transfer to normal distribution (log)
  df$avhv_log <- log(df$avhv)
  df$incm_log <- log(df$incm)
  df$inca_log <- log(df$inca)
  df$rgif_log <- log(df$rgif)
  df$agif_log <- log(df$agif)
  # Dropping 
  DropVs <- names(df) %in% c("avhv", "incm", "inca","rgif","agif") 
  df <- df[!DropVs]
  return(df)
}

# Train.clean <- dfcleaning(train)
# summary(Train.clean )

charity<- read.csv(file='/Users/liyuze/Desktop/charity.csv')
# data cleaning and transformation for charity data
charity.t <- dfcleaning(charity)
# Order column names

charity.t<- subset(charity.t, select=c(ID:tlag,chld_0:agif_log,donr:part))

# set up data for analysis
data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,2:27]
c.train <- data.train[,28] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,29] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,2:27]
c.valid <- data.valid[,28] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,29] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,2:27]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)


#====================================================================================
# CLASSIFICATION MODELING 
#====================================================================================

#### Logistic Regression
library(ISLR)
library(leaps)
library(MASS)
library(tidyverse)
library(caret)
library(lattice)

# Full model
glm.full <- glm(as.factor(donr) ~.+ I(hinc^2)+npro*tgif, data = data.train.std.c,family=binomial)
summary(glm.full)

# Stepwise selection
step.glm <- glm.full %>% stepAIC(trace = FALSE)
coef(step.glm)

# Best Stepwise logistic
glm.stepwise <- glm(as.factor(donr) ~reg1+reg2+home+chld+wrat+npro
                    +tgif+tdon+tlag+chld_0+hinc_4+wrat_0_4+incm_log+I(hinc^2)+npro*tgif, 
                    data = data.train.std.c,family=binomial)

# Check VIF : checking multicollinearity
library(car)
vif(glm.stepwise) # All below 4

# Final besy logistic regession model
summary(glm.stepwise)

# Test on valid
valid.glm <-predict(glm.stepwise,data.valid.std.c) 

profit.glm <- cumsum(14.5*c.valid[order(valid.glm, decreasing=T)]-2)
plot(profit.glm, main = "glm Plot") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.glm) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.glm)) # report number of mailings and maximum profit
# 1291.0 11816.5

cutoff.glm <- sort(valid.glm, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.glm <- ifelse(valid.glm>cutoff.glm, 1, 0) # mail to everyone above the cutoff
table(chat.valid.glm, c.valid) # classification table

### GAM Logistic Model

library(gam)

gam.full <- gam(donr~.,data.train.std.c, family = "gaussian") 
summary(gam.full)
gam.lr <- gam(donr~reg1+reg2+home+chld+hinc+wrat+plow+npro+tgif+
                tdon+tlag+chld_0+hinc_4+wrat_0_4+avhv_log+incm_log+
                I(hinc^2)+npro*tgif ,
              data.train.std.c, family = "gaussian")

summary(gam.lr)
post.valid.gam.lr<-predict(gam.lr,data.valid.std.c) # n.valid post probs

profit.gam.lr <- cumsum(14.5*c.valid[order(post.valid.gam.lr, decreasing=T)]-2)
plot(profit.gam.lr, main = "GAM.LR Plot") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.gam.lr) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.gam.lr)) # report number of mailings and maximum profit
# 1291.0 11816.5

cutoff.gam.lr <- sort(post.valid.gam.lr, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.gam.lr <- ifelse(post.valid.gam.lr>cutoff.gam.lr, 1, 0) # mail to everyone above the cutoff
table(chat.valid.gam.lr, c.valid) # classification table


###LDA Model

library(MASS)
model.lda.full <- lda(donr ~.+I(hinc^2)+npro*tgif, data.train.std.c)
summary(model.lda.full)

model.lda <- lda(donr ~reg1 + reg2 + hinc + I(hinc^2) +hinc_4+incm_log+tgif+
                   wrat + npro + tdon + tlag+ chld_0+home+home*chld +incm_log*plow +npro*tgif, data.train.std.c)
summary(model.lda)

valid.lda <- predict(model.lda, data.valid.std.c)$posterior[,2] # n.valid.c post probs

profit.lda <- cumsum(14.5*c.valid[order(valid.lda, decreasing=T)]-2)
plot(profit.lda, main = "LDA Plot") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda)) # report number of mailings and maximum profit
#1279.0 11782.5
cutoff.lda <- sort(valid.lda, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda <- ifelse(valid.lda>cutoff.lda, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda, c.valid) # classification table


###QDA Model

library(MASS)
model.qda.full <- qda(donr ~.+I(hinc^2)+npro*tgif, data.train.std.c)
summary(model.qda.full)

model.qda <- qda(donr ~reg1 + reg2 + hinc + I(hinc^2) +hinc_4+incm_log+tgif+
                   wrat + npro + tdon + tlag+ chld_0+home+
                   home*chld +incm_log*plow +npro*tgif, data.train.std.c)
summary(model.qda)

valid.qda <- predict(model.qda, data.valid.std.c)$posterior[,2] # n.valid.c post probs
profit.qda <- cumsum(14.5*c.valid[order(valid.qda, decreasing=T)]-2)
plot(profit.qda, main = "QDA Plot") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.qda) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.qda)) # report number of mailings and maximum profit
#1229 11404
cutoff.qda <- sort(valid.qda, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.qda <- ifelse(valid.qda>cutoff.qda, 1, 0) # mail to everyone above the cutoff
table(chat.valid.qda, c.valid) # classification table


###KNN Model

library(class)
KNNVars <- c("reg1","reg2","reg3","reg4","home","chld", "hinc", "genf","wrat","plow","npro",
             "tgif","tdon","tlag","avhv_log","incm_log","inca_log","rgif_log","rgif_log")

post.valid.knn<-knn(data.train.std.c[,KNNVars],data.valid.std.c[,KNNVars],c.train,k = 5)

profit.knn <- cumsum(14.5*c.valid[order(post.valid.knn, decreasing=T)]-2)
plot(profit.knn, main = "KNN Plot") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.knn) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn)) # report number of mailings and maximum profit

cutoff.knn <- sort(post.valid.knn, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
table(post.valid.knn, c.valid) # classification table
knnError <- mean(post.valid.knn != c.valid)
knnError

#### SVM

library(e1071)
SVM.tune <- tune(svm, as.factor(donr)~.+ I(hinc^2), 
                 data = data.train.std.c,
                 kernal="linear",
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

summary(SVM.tune)
SVM.best <- svm(as.factor(donr)~.+ I(hinc^2), data = data.train.std.c,kernal="linear",cost=1, scale = FALSE)
summary(SVM.best)

# Test on valid
valid.svm <-predict(SVM.best,data.valid.std.c) 
profit.svm <- cumsum(14.5*c.valid[order(valid.svm, decreasing=T)]-2)
plot(profit.svm, main = "glm Plot") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.svm) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.svm)) # report number of mailings and maximum profit
# 1060 11336


table(valid.svm, c.valid)
# Error 0.1625372
svmError <- mean(valid.svm!= c.valid)
svmError

### Decision Tree

# Classification Tree with rpart
library(rpart)
TreeModel <- rpart(as.factor(donr)~.+ I(hinc^2), 
                   method="class", 
                   control=rpart.control(minsplit=100, cp=0.01),
                   data=data.train.std.c)

printcp(TreeModel) # display the results 
plotcp(TreeModel) # visualize cross-validation results 
summary(TreeModel) # detailed summary of splits



# Test on valid
valid.Tree <-predict(TreeModel,data.valid.std.c,type = "class")

profit.Tree <- cumsum(14.5*c.valid[order(valid.Tree, decreasing=T)]-2)
plot(profit.Tree, main = "Tree Plot") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.Tree) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.Tree)) # report number of mailings and maximum profit
# 1106.0 10649.5


table(valid.Tree, c.valid)
# Error 0.1625372
TreeError <- mean(valid.Tree!= c.valid)
TreeError

### Random Forest
library(randomForest)
set.seed(1)

RF.Model <- randomForest(as.factor(donr)~.+ I(hinc^2), data=data.train.std.c, ntree=200, do.trace=TRUE)
plot(RF.Model)

# Varible importance
VimP_rf <- varImp(RF.Model, scale=FALSE)
#plot(imp_rf, top=20)
VimP_rf[, "Variable"] <- rownames(VimP_rf)
VimP_rf <- VimP_rf[with(VimP_rf, order(-Overall)), ]


#transform(VimP_rf, Variable=reorder(Variable, -Overall))
ggplot(transform(VimP_rf, Variable=reorder(Variable, -Overall))[1:20, ], aes(Variable, Overall)) +
  geom_bar(stat="identity", fill="blue") +
  scale_fill_manual() +
  labs(x="",
       y="Random Forest Attribute Importance") +
  theme(text=element_text(size=12), axis.text.x=element_text(angle=270, hjust=0))

# Test on valid
valid.RF <-predict(RF.Model,newdata=data.valid.std.c)

profit.RF <- cumsum(14.5*c.valid[order(valid.RF, decreasing=T)]-2)
plot(profit.RF, main = "RF Plot") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.RF) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.RF)) # report number of mailings and maximum profit
# 1106.0 10649.5

table(valid.RF, c.valid)
# Error 0.1625372
RF.Error <- mean(valid.RF!= c.valid)
RF.Error

### Boosting

library(caret)
fitControl <- trainControl(method="repeatedcv", number=5, repeats=5)
set.seed(1)

gbm.full <- train(as.factor(donr) ~reg1 + reg2 + home + chld + wrat + 
                    npro + tgif + tdon + tlag + chld_0 + hinc_4 + wrat_0_4 + 
                    incm_log + I(hinc^2) + npro * tgif, data = data.train.std.c,
                  method = "gbm", trControl = fitControl, verbose = FALSE)
gbm.full 
# The final values used for the model were n.trees = 150, 
#interaction.depth = 3, shrinkage =0.1 and n.minobsinnode = 10.

boost.model = gbm::(donr ~ reg1 + reg2 + home + chld + wrat +npro + tgif + tdon + tlag + chld_0 + hinc_4 + wrat_0_4 +incm_log + I(hinc^2) + npro * tgif, 
                  data = data.train.std.c, distribution = "bernoulli", 
                  shrinkage = 0.1, n.minobsinnode = 10, n.trees = 150, interaction.depth = 3)
?gbm
summary(boost.model)
set.seed(1)
boost.prob.model = predict.gbm(boost.model, newdata = data.valid.std.c, n.trees = 150, type = "response")
boost.pred.model = rep("0", 2018)
boost.pred.model[boost.prob.model> .5] = "1"
table(boost.pred.model , c.valid) # correct prediction 900+947/2081
boost.err.model <- mean(boost.pred.model != c.valid)
boost.err.model # [1] 0.0877106

profit.boost <- cumsum(14.5*c.valid[order(boost.prob.model , decreasing=T)]-2)
# see how profits change as more mailings are made
plot(profit.boost, main = "Maximum Profit - Boosting") 
# number of mailings that maximizes profits
n.mail.boost <- which.max(profit.boost) 
# report number of mailings and maximum profit
c(n.mail.boost, max(profit.boost)) # 1240.0 11947.5

#cutoffs
cutoff.boost <- sort(boost.prob.model, decreasing=T)[n.mail.boost+1] # set cutoff based on number of mailings for max profit
chat.boost <- ifelse(boost.prob.model  > cutoff.boost, 1, 0) # mail to everyone above the cutoff
table(chat.boost, c.valid) # classification table

#====================================================================================
# PREDICTION MODELING - for donation amount
#====================================================================================

### Least Square Regression -full

# Fit a Least Squares Regression Model using all ten predictors

lm.full<- lm(damt ~ .+I(hinc^2) + npro * tgif+inca_log*plow, 
             data = data.train.std.y) # Fit the model using the TRAINING data set
summary(lm.full) # Summary of the linear regression model

pred.full <- predict(lm.full, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.full)^2) 
# 1.542719
sd((y.valid - pred.full)^2)/sqrt(n.valid.y) # std error
# 0.1557496


### Best Subset Selection using BIC 
regfit.full <- regsubsets(damt ~ .+I(hinc^2) + npro * tgif+inca_log*plow,  
                          data = data.train.std.y, nvmax = 20)
summary(regfit.full)
par(mfrow = c(1, 2))
plot(regfit.full, scale = "bic", main = "Predictor Variables vs. BIC")
reg.summary <- summary(regfit.full)

par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(20,reg.summary$adjr2[20], col="red",cex=2,pch=20)
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(19,reg.summary$cp[19],col="red",cex=2,pch=20)
which.min(reg.summary$bic)
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(15,reg.summary$bic[15],col="red",cex=2,pch=20)
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")

coef(regfit.full, 15)

lm.BIC<- lm(damt ~ reg3+reg4+home+chld+hinc+wrat+plow+tgif+wrat_0_4+
              incm_log+rgif_log+agif_log+I(hinc^2) +(inca_log:plow)+(npro:tgif), 
            data = data.train.std.y) # Fit the model using the TRAINING data set
summary(lm.BIC) # Summary of the linear regression model

pred.bic <- predict(lm.BIC, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.bic)^2) 
# 1.543577
sd((y.valid - pred.bic)^2)/sqrt(n.valid.y) # std error
# 0.1554454


### Best Subset -10 Fold CV

k <- 10
set.seed(1306)
folds <- sample(1:k, nrow(data.train.std.y), replace = TRUE)
cv.errors <- matrix(NA, k, 10, dimnames = list(NULL, paste(1:10)))

# Let's write our own predict method
predict.regsubsets <- function(object, newdata, id,...){
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars]%*%coefi
}

for (j in 1:k) {
  best.fit <- regsubsets(damt ~ . + I(hinc^2) + npro*tgif +inca_log*plow, 
                         data = data.train.std.y[folds != j, ], nvmax = 10)
  for (i in 1:10) {
    pred <- predict(best.fit, data.train.std.y[folds == j, ], id = i)
    cv.errors[j, i] = mean((data.train.std.y$damt[folds == j] - pred)^2)
  }
}

mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors

which.min(mean.cv.errors)
mean.cv.errors[10]

par(mfrow = c(1,2))
plot(mean.cv.errors, type = 'b', xlab = "Number of Predictors", ylab = "Mean CV Errors",
     main = "Best Subset Selection (10-fold CV)")
points(10, mean.cv.errors[10], col = "brown", cex = 2, pch = 20)
rmse.cv = sqrt(apply(cv.errors, 2, mean))
rmse.cv[10]
plot(rmse.cv, pch = 19, type = "b", xlab = "Number of Predictors", ylab = "RMSE CV",
     main = "Best Subset Selection (10-fold CV)")
points(10, rmse.cv[10], col = "blue", cex = 2, pch = 20)

reg.best.CV <- regsubsets(damt ~ . + I(hinc^2) + npro*tgif +inca_log*plow, 
                          data = data.train.std.y, nvmax = 20)
coef(reg.best.CV, 10)

lm.10cv <- lm(damt ~ reg3+reg4+chld+hinc+wrat+tgif+wrat_0_4+
                rgif_log+agif_log+inca_log*plow, 
              data = data.train.std.y) # Fit the model using the TRAINING data set
summary(lm.10cv) # Summary of the linear regression model


pred.10cv<- predict(lm.10cv, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.10cv)^2) 
# 1.614235
sd((y.valid - pred.10cv)^2)/sqrt(n.valid.y) # std error
#  0.1581979


### Ridge

## Ridge regression model using 10-fold cross-validation to select that largest
# value of lambda s.t. the CV error is within 1 s.e. of the minimum

library(glmnet)
x <- model.matrix(damt~., data.train.std.y)[,-1]  
y <- data.train.std.y[,27]

# Validation
x.val <- model.matrix(damt~., data.valid.std.y)[,-1]
y.val <- data.valid.std.y[,27]
grid=10^seq(10,-2,length=100)


par(mfrow = c(1,2))
grid <- 10^seq(10, -2, length = 100)
ridge.mod <- glmnet(x,y, alpha = 0, lambda = grid, thresh = 1e-12)
plot(ridge.mod, xvar = "lambda", label = TRUE)
set.seed(1306)
cv.out <- cv.glmnet(x,y, alpha = 0)
plot(cv.out)

bestlam <- cv.out$lambda.min
bestlam # Lambda = 0.1226929 (leads to smallest CV error)
log(bestlam) #-2.098071

ridge.mod <- glmnet(x, y, alpha = 0, lambda = bestlam)
ridge.pred <- predict(ridge.mod, s = bestlam, newx = x.val)
mean((ridge.pred - y.val)^2) # Mean Prediction Error = 3074.378
# 1.60079
sd((y.val - ridge.pred)^2)/sqrt(n.valid.y) # std error
# 0.1581979


### Lasso 

par(mfrow = c(1,2))
grid <- 10^seq(10, -2, length = 100)
lasso.mod <- glmnet(x, y, alpha = 1, lambda = grid, thresh = 1e-12)
plot(lasso.mod, xvar = "lambda", label = TRUE)

set.seed(1306)
cv.out <- cv.glmnet(x, y, alpha = 1)
plot(cv.out)

bestlam <- cv.out$lambda.min
bestlam # Lambda = 0.005069684 (leads to smallest CV error)

lasso.mod <- glmnet(x, y, alpha = 0, lambda = bestlam)
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x.val)
mean((lasso.pred - y.val)^2) # Mean Prediction Error = 3074.378
# 1.581835
sd((y.val - lasso.pred)^2)/sqrt(n.valid.y) 
# std error: 0.1556258


### Principal Components Analysis
library(pls)
set.seed(1)
pcr.fit=pcr(damt~., data=data.train.std.y , scale=FALSE , validation="CV")
summary(pcr.fit)
validationplot(pcr.fit,val.type="MSEP")
set.seed(1)
pcr.fit=pcr(damt~., data=data.train.std.y , scale=FALSE , validation="CV")
validationplot(pcr.fit,val.type="MSEP")
pcr.pred=predict(pcr.fit, data.valid.std.y,ncomp=22)
mean((y.valid-pcr.pred)^2)
sd((y.valid- pcr.pred)^2)/sqrt(n.valid.y)
summary(pcr.fit)

### Partial Least Squares
set.seed(1)
pls.fit=plsr(damt~., data=data.train.std.y ,scale=TRUE,ncomp=4)
summary(pls.fit)
validationplot(pls.fit,val.type="MSEP")

pls.pred=predict(pls.fit,data.valid.std.y,ncomp=4)
mean((y.valid-pls.pred)^2)
sd((y.valid- pls.pred)^2)/sqrt(n.valid.y)


### Random Forests
set.seed(1)
RF.pred <- randomForest(damt ~ agif_log+rgif_log+reg4+lgif+ResponseRate+chld+
                          GiftTimes+chld_0+hinc+tgif+plow+incm_log+reg3+
                          reg2+wrat+npro+inca_log,data=data.train.std.y, importance=TRUE,ntree=100, do.trace=TRUE)
plot(RF.pred)
# varaible importance
varImp(RF.pred, scale=FALSE)
RF.pred.amt <- predict(RF.pred, newdata = data.valid.std.y)
mean((y.valid-RF.pred.amt)^2)
sd((y.valid- RF.pred.amt)^2)/sqrt(n.valid.y)


### Boosting
set.seed(1)
boost.pred <- gbm(damt ~., data = data.train.std.y, distribution = "gaussian",
                  n.trees=5000, interaction.depth=4)
summary(boost.pred)

set.seed(1)
boost.pred <- gbm(damt ~rgif_log+agif_log+reg4+chld+lgif+hinc+wrat+reg3+
                    chld_0+incm_log+plow+reg2+tgif+chld_ThM+inca_log+avhv_log+
                    home+ResponseRate+npro+tdon+reg1+GiftTimes, 
                  data = data.train.std.y, distribution = "gaussian",
                  n.trees=5000, interaction.depth=5)
summary(boost.pred)
boost.pred.amt  <- predict(boost.pred, newdata = data.valid.std.y, n.trees = 5000)

mean((y.valid-boost.pred.amt)^2)
sd((y.valid- boost.pred.amt)^2)/sqrt(n.valid.y)

#====================================================================================
# Apply the best classification and prediction mode into test data set
#====================================================================================

### Classification ###
# select boost.model since it has maximum profit in the validation sample
# post probs for test data

post.test <- predict.gbm(boost.model, newdata = data.test.std, n.trees = 150, type = "response") 

# Oversampling adjustment for calculating number of mailings for test set

n.mail.valid <- which.max(profit.boost)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)

### Prediction ####
# select boosting since it has minimum mean prediction error in the validation sample

yhat.test <- predict(boost.pred, 
                     newdata = data.test.std, 
                     n.trees = 5000) # test predictions

# FINAL RESULTS

# Save final results for both classification and regression

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="testresult.csv", row.names=FALSE) # use your initials for the file name
summary(ip)
summary(subset(ip, ip$chat==1))
# submit the csv file in Canvas for evaluation based on actual test donr and damt values