PRACTICAL MACHINE LEARNING - Course Project
===========================================
*"How did they do the exercise ?"*

by Javier Martínez Arribas

## Loading and preprocessing the data
We can get the data from:

[Training Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
[Testing data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

The goal of this analysis is to predict the manner in which the subjects did the 
recorded exercise. For this purpose we have got two data sets, one training set 
and one testing set to apply our machine learning algorithm to twenty test cases.

We will split the training set into a train and a test set to make our model.
First of all we get the files from the web:


```r
library(caret)
#Training Set
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
dest <- getwd()
dest <- paste(dest,sep="","/pml-training.csv")
download.file(url,destfile=dest,method="curl")
training <- read.csv(dest,header=T)

#Test Set
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
dest <- getwd()
dest <- paste(dest,sep="","/pml-testing.csv")
download.file(url,destfile=dest,method="curl")
testing <- read.csv(dest,header=T)
```

**Pre processing**

In order to address the problem of which predictors are better to construct the 
model we are going to make a deconstruction of the training set: 
First we separate the outcome from predictors. Outcome is the last column(classe):

```r
Y <- training[,160]
X <- training[,-160]
```

Then we can see what classes of predictors there are:

```r
table(sapply(X[1, ], class))
```

```
## 
##  factor integer numeric 
##      36      35      88
```

Once we have the types, we could divide the predictors set in order to explore 
deeply their importance for the final model.

```r
int <- sapply(X[1, ], class)=="integer"
X_int <- X[,int]
num <- sapply(X[1, ], class)=="numeric"
X_num <- X[,num]
fact <- sapply(X[1, ], class)=="factor"
X_fact <- X[,fact]
```
 
For the first subset formed by integer predictors we can considerate to remove 
some of them like:

```r
str(X_int)
```

```
## 'data.frame':	19622 obs. of  35 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ max_picth_belt      : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt      : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt: int  NA NA NA NA NA NA NA NA NA NA ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ max_yaw_arm         : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm         : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm   : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
```
The first predictor is just a counter of the rows in data set, so we can remove it:

```r
head(X_int$X)
```

```
## [1] 1 2 3 4 5 6
```

```r
tail(X_int$X)
```

```
## [1] 19617 19618 19619 19620 19621 19622
```
Now, we can examinate the predictors with too much NAs at first glance:

```r
dim(X_int[complete.cases(X_int$max_picth_belt),])[1]
```

```
## [1] 406
```

```r
dim(X_int[complete.cases(X_int$min_pitch_belt),])[1]
```

```
## [1] 406
```

```r
dim(X_int[complete.cases(X_int$amplitude_pitch_belt),])[1]
```

```
## [1] 406
```

```r
dim(X_int[complete.cases(X_int$max_yaw_arm),])[1]
```

```
## [1] 406
```

```r
dim(X_int[complete.cases(X_int$min_yaw_arm),])[1]
```

```
## [1] 406
```

```r
dim(X_int[complete.cases(X_int$amplitude_yaw_arm),])[1]
```

```
## [1] 406
```

```r
dim(X_int)[1]
```

```
## [1] 19622
```
We discover that there are 406 complete cases out of 19622 in these predictors,
almost the 4%, so we could assume that there is not much relevant information into 
so we remove them as well.
We can remove also the time_stamp variables because we need to predict the way 
the exercise is done and there is no need to know the execution time.

```r
X_int <- subset(X_int,select=-c(X,raw_timestamp_part_1,raw_timestamp_part_2,num_window,
                                max_picth_belt,min_pitch_belt,amplitude_pitch_belt
                                      ,max_yaw_arm,min_yaw_arm,amplitude_yaw_arm))
dim(X_int[complete.cases(X_int),])[1]
```

```
## [1] 19622
```
As we can see now we have no NA values in this predictor subset.
We could study the correlation between predictors to avoid some redundant
information. We use the findCorrelation function in caret package.

```r
descrCor_int <- cor(X_int)
highlyCorDescr_int <- findCorrelation(descrCor_int, cutoff = 0.8)
X_int <- X_int[,-highlyCorDescr_int]
```
For the second predictors subset we are going to study numeric variables in the same way 
as integer ones:

```r
str(X_num)
```

```
## 'data.frame':	19622 obs. of  88 variables:
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_dumbbell: num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_accel_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_dumbbell    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_dumbbell   : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_dumbbell     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_dumbbell_x        : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y        : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z        : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ magnet_dumbbell_z       : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm            : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm           : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm             : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ max_roll_forearm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_forearm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_forearm  : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_forearm : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_accel_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_forearm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_forearm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_forearm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_forearm    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_forearm       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_forearm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_forearm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_forearm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_forearm_x         : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y         : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z         : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ magnet_forearm_y        : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z        : num  476 473 469 469 473 478 470 474 476 473 ...
```
Now we have got the same problem as with the integer variables there are some of
them with just 406 valid samples out of 19622, so we should remove them.

```r
for (i in 1:ncol(X_num)){
  if (i==1){nAs=NULL}
  nAs <- c(nAs,dim(X_num[complete.cases(X_num[,i]),])[1])
}
table(nAs)
```

```
## nAs
##   406 19622 
##    61    27
```
and to remove just that columns...

```r
for (i in 1:ncol(X_num)){
  if (i==1){nam <- NULL}
  if ((dim(X_num[complete.cases(X_num[,i]),])[1]) == 406) {
    nam <- c(nam,colnames(X_num)[i])
  }
}
indices <- which(names(X_num) %in% nam)
X_num <- X_num[,-indices]
```
We can study the correlation between predictors in this subset as well:

```r
descrCor_num <- cor(X_num)
highlyCorDescr_num <- findCorrelation(descrCor_num, cutoff = 0.75)
X_num <- X_num[,-highlyCorDescr_num]
```

And the last step in order to get the most appropriate information for our model,
we study the most relevant factor predictors.
In this subset we have different cases to address to, like character variables in
factor format as well as date variables. We have also some factor variables with 
two levels, one of them "" and the other one "#DIV/0!", a kind of notation for 
error values in excel files. 
We have some numeric variables format as factor, categorical and binary ones.
I consider that we should not take into account this subset of predictors due to 
the few information contained into.
Just in order to show the huge number of zero values we are going to transform 
factor variables into numeric ones after removing some of no sense predictors.

```r
X_fact <- subset(X_fact,select=-c(new_window,user_name,cvtd_timestamp,kurtosis_yaw_belt,
                                         skewness_yaw_belt,amplitude_yaw_belt,
                                         kurtosis_yaw_dumbbell,skewness_yaw_dumbbell,
                                         amplitude_yaw_dumbbell,kurtosis_yaw_forearm,
                                         skewness_yaw_forearm,amplitude_yaw_forearm))
```
and transform the rest in numeric variables:

```r
for(i in 1:ncol(X_fact)){
  #Take the index where there are negative values
  ind <- grep("-",X_fact[,i])
  X_fact[,i] <- gsub("-","",X_fact[,i])
  X_fact[,i] <- invisible(as.numeric(X_fact[,i]))
  bad <- is.na(X_fact[,i])
  X_fact[bad,i] <- 0
  X_fact[ind,i]=X_fact[ind,i]*(-1)
}
head(X_fact)
```

```
##   kurtosis_roll_belt kurtosis_picth_belt skewness_roll_belt
## 1                  0                   0                  0
## 2                  0                   0                  0
## 3                  0                   0                  0
## 4                  0                   0                  0
## 5                  0                   0                  0
## 6                  0                   0                  0
##   skewness_roll_belt.1 max_yaw_belt min_yaw_belt kurtosis_roll_arm
## 1                    0            0            0                 0
## 2                    0            0            0                 0
## 3                    0            0            0                 0
## 4                    0            0            0                 0
## 5                    0            0            0                 0
## 6                    0            0            0                 0
##   kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm
## 1                  0                0                 0                  0
## 2                  0                0                 0                  0
## 3                  0                0                 0                  0
## 4                  0                0                 0                  0
## 5                  0                0                 0                  0
## 6                  0                0                 0                  0
##   skewness_yaw_arm kurtosis_roll_dumbbell kurtosis_picth_dumbbell
## 1                0                      0                       0
## 2                0                      0                       0
## 3                0                      0                       0
## 4                0                      0                       0
## 5                0                      0                       0
## 6                0                      0                       0
##   skewness_roll_dumbbell skewness_pitch_dumbbell max_yaw_dumbbell
## 1                      0                       0                0
## 2                      0                       0                0
## 3                      0                       0                0
## 4                      0                       0                0
## 5                      0                       0                0
## 6                      0                       0                0
##   min_yaw_dumbbell kurtosis_roll_forearm kurtosis_picth_forearm
## 1                0                     0                      0
## 2                0                     0                      0
## 3                0                     0                      0
## 4                0                     0                      0
## 5                0                     0                      0
## 6                0                     0                      0
##   skewness_roll_forearm skewness_pitch_forearm max_yaw_forearm
## 1                     0                      0               0
## 2                     0                      0               0
## 3                     0                      0               0
## 4                     0                      0               0
## 5                     0                      0               0
## 6                     0                      0               0
##   min_yaw_forearm
## 1               0
## 2               0
## 3               0
## 4               0
## 5               0
## 6               0
```
We can see that a large number of rows are zero values.

```r
row_sub = apply(X_fact, 1, function(row) all(row ==0 ))
table(row_sub)
```

```
## row_sub
## FALSE  TRUE 
##   406 19216
```
There are just 406 no zero rows out of 12622 rows. Almost a 4%
of the entire dataset.
After studying the different types of predictors we concatenate the predictors 
subsets together. Discarding factor transformed predictors:

```r
training_final <- cbind(Y,X_int,X_num)
```

Now we can define the control structure for training:

```r
set.seed(123)
ind <- createFolds(Y,returnTrain=TRUE)
cont <- trainControl(method="cv",index=ind)
```
For tune the desired model and calculate the accuracy of it we split the 
training set into a train and a test subset.
We fit a Random Forest method for this classification problem. In order not to waste
too much time nor computation resources we will select just ten trees and even so
we will get good predictions for our final exercise.
I did try to compute it with a higher number of trees in random forest but final 
predictions did not change.

```r
set.seed(123)
mtryVals <- floor(seq(10,ncol(training_final[,-1]),length=10))
mtryGrid <- data.frame(.mtry=mtryVals)
inTrain <- createDataPartition(training_final$Y,p = .8, list = FALSE)
train <- training_final[inTrain,]
test <- training_final[-inTrain,]
rfTune <- train(x = training_final[,-1], y = training_final[,1], method = "rf", 
                tuneGrid=mtryGrid,ntree=10, importance=TRUE, trControl = cont)
pred_train <- predict(rfTune,train)
ISerror <- sum(pred_train != train$Y) * 100 / nrow(train)
ISerror
```

```
## [1] 0.01911
```
We get an in sample error of almost 0.02%.
and we observe an accuracy and an out of sample error of:

```r
pred <- predict(rfTune,test)
table(pred,test$Y)
```

```
##     
## pred    A    B    C    D    E
##    A 1116    1    0    0    0
##    B    0  758    0    0    0
##    C    0    0  684    1    0
##    D    0    0    0  642    0
##    E    0    0    0    0  721
```

```r
confusionMatrix(pred,test$Y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  758    0    0    0
##          C    0    0  684    1    0
##          D    0    0    0  642    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                     
##                Accuracy : 0.999     
##                  95% CI : (0.998, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 0.999     
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.999    1.000    0.998    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          0.999    1.000    0.999    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.285    0.193    0.175    0.164    0.184
## Balanced Accuracy       1.000    0.999    1.000    0.999    1.000
```

```r
OOSerror <- sum(pred != test$Y) * 100 / nrow(test)
OOSerror
```

```
## [1] 0.05098
```
We obtain a low out of sample error 0.05% and an accuracy of 99.95%.
Finally, we are going to use the testing file to complete the prediction exercise.
First we need to select just the predictors that we are going to need..

```r
indices <- which(names(testing) %in% names(train))
testing <- testing[,indices]
pred_final <- predict(rfTune,testing)
pred_final
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
Finally we get a random forest model for 43 predictors with a high accuracy and
a low out of sample error.
