# Credit_Risk_Analysis

# Overview

The purpose of this repository was to employ different machine learning techniques to train and evaluate models with unbalanced classes to assess the credit risk metrics associated with loan approval odds to identify what technique is best for LendingClub. The imbalanced-learn and scikit-learn libraries were used to build and evaluate models using resampling. The data was oversampled using RandomOverSampler and SMOTE algorithms, and undersampled using the ClusterCentroids algorithm. The SMOTEENN algorithm was used to combine over-and-under sampling of the data. Finally, bias was reduced using the BalancedRandomForestClassifier and EasyEnsembleClassifier models. 

# Results 

## Resampling

### Naive Random Sampling
![image](https://user-images.githubusercontent.com/95376544/164165531-e06ecda7-1eeb-4696-a517-1a3d225d0425.png)
![image](https://user-images.githubusercontent.com/95376544/164173979-9c3fca9a-69bc-4660-b6b4-4d0099863928.png)

The RandomOverSampler Model over-samples the minority class(es) by picking samples at random with replacement to equal the majority class. 

We can see that:
 * For low risk, the precision rate was 100%, the recall was 68% and the F1 score was 81%
 * For high risk, the precision rate was 1%, the recall was 57% and the F1 score was 2%. 
 * The balanced accuracy score was 62.9%.

### SMOTE Oversampling
![image](https://user-images.githubusercontent.com/95376544/164167960-7cc43b3d-195b-44eb-8fec-653a9c9b52da.png)
![image](https://user-images.githubusercontent.com/95376544/164174026-876670df-56cc-455b-9c05-4776a888feb9.png)

 In SMOTE, like random oversampling, the size of the minority is increased with new instance interpolation using nearby datapoints.

 We can see that:
 * For low risk, the precision rate was 100%, the recall was 63% and the F1 score was 78%
 * For high risk, the precision rate was 1%, the recall was 62% and the F1 score was 2%. 
 * The balanced accuracy score was 62.7%.
 * A drawback of this technique is that new data points created can be heavily influenced by outliers.


### Undersampling 
![image](https://user-images.githubusercontent.com/95376544/164170558-6f06f6e4-0f36-4104-b9b2-807f00e5f0eb.png)
![image](https://user-images.githubusercontent.com/95376544/164174068-7bdfabd1-842a-42d4-8167-8495fa7f9aff.png)

This is a method that under samples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm. (https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html)

We can see that:
 * For low risk, the precision rate was 100%, the recall was 61% and the F1 score was 73%
 * For high risk, the precision rate was 1%, the recall was 57% and the F1 score was 1%. 
 * The balanced accuracy score was 59%.


### Combination (Over and Under) Sampling using SMOTEENN
![image](https://user-images.githubusercontent.com/95376544/164178604-fe18750f-5e77-4591-b0f9-f7d297973e1a.png)
![image](https://user-images.githubusercontent.com/95376544/164174115-1baac7dd-8e90-4fac-bb86-fa59c3467ff2.png)

SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms to oversample the minority class with SMOTE and cleans the resulting data with an undersampling strategory by dropping data points if its two nearest neighbors belong to different classes. 

We can see that:
 * For low risk, the precision rate was 100%, the recall was 61% and the F1 score was 76%
 * For high risk, the precision rate was 1%, the recall was 70% and the F1 score was 2%. 
 * The balanced accuracy score was 59%.

## Ensemble Learners

### Balanced Random Forest Classifier
![image](https://user-images.githubusercontent.com/95376544/164172094-692b1f4b-803d-40ba-9e2d-1695d553c3d1.png)
![image](https://user-images.githubusercontent.com/95376544/164173837-6361f8e1-3bd2-4cf0-aeae-d5e2f29788f0.png)

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

We can see that:
 * For low risk, the precision rate was 100%, the recall was 67% and the F1 score was 95%
 * For high risk, the precision rate was 4%, the recall was 91% and the F1 score was 7%. 
 * The balanced accuracy score was 78.7%

### Easy Ensemble AdaBoost Classifier
![image](https://user-images.githubusercontent.com/95376544/164172618-25957ead-a885-4955-b72b-076f51406f67.png)
![image](https://user-images.githubusercontent.com/95376544/164173886-d8b4e9f3-2706-4f8b-a4e9-76c42b8a2c0c.png)

An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

We can see that:
 * For low risk, the precision rate was 100%, the recall was 91% and the F1 score was 97%
 * For high risk, the precision rate was 7%, the recall was 94% and the F1 score was 14%. 
 * The balanced accuracy score was 92.5%.
 * This is the most accurate, strongest supported method.

## Summary

Looking at the balanced accuracy scores, recalls, and F1 scores, we can see that the least effective methods with marginal differences were the SMOTEENN (Over and Under Sampling) and the Undersampling methods. Out of the resampling methods, Random Oversampling appeared to be the strongest. However, the Ensemble Learners were the most accurate, seeing a significant increase among all scores. I would suggest using the Easy Ensemble AdaBoost Classifier since it has the highest scores out of all of the methods used. 
