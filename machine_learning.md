---
title: "Analyzing Fitness Data"
author: "Liran Bareket"
date: "Friday, April 24, 2015"
output: html_document
---
## Coursera - Practical Machine Learning 
### Course Submission.

These are the steps we used to analyze the data.  
We read the data into a data frame.  
We set the classes for the numeric columns we would like to analyze.  
These are the raw data elements (no aggregates).  
We then removed all the columns we are not going to use from the data frame.  


```r
library(caret)
library(knitr)
# We are setting the columns we want to use to numeric.
colClasses<-c(rep("character",7),
                                 rep("numeric",4),
                                 rep("character",25),
                                 rep("numeric",13),
                                 rep("character",10),
                                 rep("numeric",9),
                                 rep("character",15),
                                 rep("numeric",3),
                                 rep("character",15),
                                 rep("numeric",1),
                                 rep("character",10),
                                 rep("numeric",12),
                                 rep("character",15),
                                 rep("numeric",1),
                                 rep("character",10),
                                 rep("numeric",9),
                                 "factor")
dataset<-read.csv("../pml-training.csv",colClasses=colClasses,na.strings=c("\"\""))
dataset_slim<-dataset[,lapply(dataset,class)=="numeric"]
dataset_slim$classe<-dataset$classe
```
Next we find the redundant or highly correlated variable and take them out of the data.  
We use the cor function to find the data correlations and then filter out all the ones that show correlation of more than .75 to other features.

```r
correlationMatrix <- cor(dataset_slim[,1:51])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
names(dataset_slim)[highlyCorrelated]
```

```
##  [1] "accel_belt_z"      "roll_belt"         "accel_belt_y"     
##  [4] "accel_arm_y"       "total_accel_belt"  "accel_belt_x"     
##  [7] "accel_dumbbell_z"  "pitch_belt"        "magnet_dumbbell_x"
## [10] "accel_dumbbell_y"  "magnet_dumbbell_y" "accel_arm_x"      
## [13] "accel_dumbbell_x"  "accel_arm_z"       "magnet_arm_y"     
## [16] "magnet_belt_y"     "accel_forearm_y"   "gyros_arm_y"      
## [19] "gyros_forearm_z"   "gyros_dumbbell_x"
```


Then we remove the redundant features from the set and split it to training and testing set.

```r
dataset_slim<-dataset_slim[,-highlyCorrelated]

##splitting to training and testing data
trainIndex <- createDataPartition(dataset_slim$classe, p=.8, list=F)
training<-dataset_slim[trainIndex,]
testing<-dataset_slim[-trainIndex,]

head(dataset_slim)
```

```
##   yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z magnet_belt_x
## 1    -94.4         0.00         0.00        -0.02            -3
## 2    -94.4         0.02         0.00        -0.02            -7
## 3    -94.4         0.00         0.00        -0.02            -2
## 4    -94.4         0.02         0.00        -0.03            -6
## 5    -94.4         0.02         0.02        -0.02            -6
## 6    -94.4         0.02         0.00        -0.02             0
##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm gyros_arm_x
## 1          -313     -128      22.5    -161              34        0.00
## 2          -311     -128      22.5    -161              34        0.02
## 3          -305     -128      22.5    -161              34        0.02
## 4          -310     -128      22.1    -161              34        0.02
## 5          -302     -128      22.1    -161              34        0.00
## 6          -312     -128      22.0    -161              34        0.02
##   gyros_arm_z magnet_arm_x magnet_arm_z roll_dumbbell pitch_dumbbell
## 1       -0.02         -368          516      13.05217      -70.49400
## 2       -0.02         -369          513      13.13074      -70.63751
## 3       -0.02         -368          513      12.85075      -70.27812
## 4        0.02         -372          512      13.43120      -70.39379
## 5        0.00         -374          506      13.37872      -70.42856
## 6        0.00         -369          513      13.38246      -70.81759
##   yaw_dumbbell total_accel_dumbbell gyros_dumbbell_y gyros_dumbbell_z
## 1    -84.87394                   37            -0.02             0.00
## 2    -84.71065                   37            -0.02             0.00
## 3    -85.14078                   37            -0.02             0.00
## 4    -84.87363                   37            -0.02            -0.02
## 5    -84.85306                   37            -0.02             0.00
## 6    -84.46500                   37            -0.02             0.00
##   magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
## 1               -65         28.4         -63.9        -153
## 2               -64         28.3         -63.9        -153
## 3               -63         28.3         -63.9        -152
## 4               -60         28.1         -63.9        -152
## 5               -68         28.0         -63.9        -152
## 6               -66         27.9         -63.9        -152
##   total_accel_forearm gyros_forearm_x gyros_forearm_y accel_forearm_x
## 1                  36            0.03            0.00             192
## 2                  36            0.02            0.00             192
## 3                  36            0.03           -0.02             196
## 4                  36            0.02           -0.02             189
## 5                  36            0.02            0.00             189
## 6                  36            0.02           -0.02             193
##   accel_forearm_z magnet_forearm_x magnet_forearm_y magnet_forearm_z
## 1            -215              -17              654              476
## 2            -216              -18              661              473
## 3            -213              -18              658              469
## 4            -214              -16              658              469
## 5            -214              -17              655              473
## 6            -215               -9              660              478
##   classe
## 1      A
## 2      A
## 3      A
## 4      A
## 5      A
## 6      A
```

We will use the random tree method to train the model. We use the random forest because it has a built in resampling mechanism.


```r
##fit the model
library(randomForest)
modFit<-randomForest(classe~. ,data=training, method="rf")
modFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, method = "rf") 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 0.62%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4459    3    0    1    1 0.001120072
## B   14 3016    6    0    2 0.007241606
## C    0   21 2701   16    0 0.013513514
## D    1    0   23 2546    3 0.010493587
## E    0    0    1    5 2880 0.002079002
```

As is apparent the OOB estimate is 0.6% which is extremely good.
We will use this model and now validate it against the testing set we prepared.


```r
##fit the model
confusionMatrix(testing$classe,predict(modFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    1    0    0    0
##          B    2  753    3    0    1
##          C    0    5  676    3    0
##          D    0    0    7  636    0
##          E    0    0    0    1  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9912, 0.9963)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9926          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9921   0.9854   0.9938   0.9986
## Specificity            0.9996   0.9981   0.9975   0.9979   0.9997
## Pos Pred Value         0.9991   0.9921   0.9883   0.9891   0.9986
## Neg Pred Value         0.9993   0.9981   0.9969   0.9988   0.9997
## Prevalence             0.2847   0.1935   0.1749   0.1631   0.1838
## Detection Rate         0.2842   0.1919   0.1723   0.1621   0.1835
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9989   0.9951   0.9915   0.9958   0.9992
```

We can see that the model predicted the classes with a very high accuracy.
We will use this model to predict the testing.
