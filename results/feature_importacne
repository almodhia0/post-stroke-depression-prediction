## Understanding Mutual Information (MI) and Adjusted Mutual Information (AMI)

In machine learning, especially when selecting the best features for a model, we need ways to measure how *related* a feature is to the thing we're trying to predict (the target variable). Mutual Information (MI) and Adjusted Mutual Information (AMI) are two such measures. They tell us how much a feature *reveals* about the target.

### Mutual Information (MI)

Think of MI as a measure of "shared information." It tells you how much *less uncertain* you are about the target variable if you know the value of a particular feature.

*   **High MI:** If knowing a feature significantly reduces your uncertainty about the target, they have high mutual information.  This suggests the feature is likely important for prediction.
*   **Low MI:** If knowing a feature doesn't change your uncertainty about the target much, they have low mutual information. This suggests the feature might not be very useful.
*   **Zero MI:** A value of zero means the feature and the target are completely independent; knowing one tells you absolutely nothing about the other.
*   **Detects All Kinds of Relationships:** MI is powerful because it can detect *any* kind of relationship, not just straight-line (linear) ones. A feature could have a complex, curvy relationship with the target, and MI would still pick it up.

### Adjusted Mutual Information (AMI)

MI has a quirk: it tends to be higher for features that have lots of different possible values (lots of categories), even if those values aren't truly more informative. This is where Adjusted Mutual Information (AMI) comes in.

*   **Corrects for Chance:** AMI is like a "fairer" version of MI. It adjusts the MI score to account for the fact that some agreement between a feature and the target could happen purely by chance.
*   **More Comparable:** Because of this adjustment, AMI is better for comparing features that have different numbers of categories.
*   **Ranges from 0 to 1 (usually):**
    *   An AMI of 1 means the feature perfectly predicts the target.
    *   An AMI of 0 means the feature is no better than random guessing.
    * (It's technically possible to get negative AMI, but that's rare and usually means something is very wrong.)

### MI and AMI in this Project

In this project, both MI and AMI help identify the most relevant factors predicting reported depression in stroke survivors. We look at both to get a comprehensive picture. AMI, in particular, gives us a more reliable comparison, ensuring that features with many categories aren't unfairly favored. The percentages shown with the AMI values are a way of normalizing the scores, showing each feature's relative importance compared to the total information from all features.

### Summary

*   **MI:**  A general measure of how related two things are. Good for finding *any* connection.
*   **AMI:**  A refined version of MI that's better for comparing features, especially if they have different numbers of categories.  Good for judging the *true strength* of a connection.

Both are valuable tools for finding the most important features for our prediction model.
