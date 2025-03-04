## Model Evaluation Metrics and Results

We evaluated the performance of four different machine learning models using several key metrics.  Here's an explanation of each metric and a summary of the results:

### Evaluation Metrics

In binary classification (where we're predicting one of two outcomes, like "reports depression" or "does not report depression"), these metrics are commonly used:

*   **Accuracy:** The overall percentage of predictions that are correct.  It's calculated as:

    ```
    Accuracy = (True Positives + True Negatives) / Total Predictions
    ```

    While easy to understand, accuracy can be misleading if the classes are imbalanced (e.g., many more people *not* reporting depression than reporting it).

*   **Precision:**  Out of all the predictions that the model labeled as *positive* (in our case, "reports depression"), what percentage were *actually* positive?  It's calculated as:

    ```
    Precision = True Positives / (True Positives + False Positives)
    ```

    High precision means the model is good at avoiding *false positives* (i.e., it doesn't incorrectly classify many people as reporting depression when they don't).

*   **Recall:** Out of all the *actually positive* cases, what percentage did the model correctly identify? It's calculated as:

    ```
    Recall = True Positives / (True Positives + False Negatives)
    ```

    High recall means the model is good at avoiding *false negatives* (i.e., it doesn't miss many people who actually report depression).

*   **F1-Score:** The harmonic mean of precision and recall. It provides a single metric that balances both precision and recall.  It's calculated as:

    ```
    F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    ```

    A high F1-score indicates good performance on both precision and recall.  It's often a better overall measure than accuracy, especially with imbalanced datasets.

*   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):**  This metric represents the model's ability to *distinguish* between the two classes (reports depression vs. does not report depression).  It's plotted as a curve (the ROC curve), and the area under that curve (AUC) is calculated.

    *   AUC-ROC = 0.5 indicates random guessing (no ability to discriminate).
    *   AUC-ROC = 1.0 indicates perfect discrimination.
    *   Values closer to 1 are better.

    AUC-ROC is less sensitive to class imbalance than accuracy.

### Model Performance Results

The following table summarizes the performance of each model on the test set:

| Model              | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| Random Forest        | 0.73     | 0.71      | 0.77   | 0.74     | 0.81    |
| Decision Tree        | 0.70     | 0.69      | 0.73   | 0.71     | 0.73    |
| Gradient Boosting    | 0.65     | 0.66      | 0.64   | 0.65     | 0.72    |
| Logistic Regression  | 0.65     | 0.65      | 0.62   | 0.64     | 0.70    |

**Interpretation:**

*   **Best Overall Model:** The **Random Forest** model achieved the best overall performance across all metrics.  It had the highest F1-score (0.74) and the highest AUC-ROC (0.81).  This indicates a good balance between precision and recall, and a strong ability to distinguish between individuals who report depression and those who do not.

*   **Random Forest Strengths:** The Random Forest model has a relatively high recall (0.77), meaning it correctly identified a large proportion of the individuals who actually reported depression.  Its precision (0.71) is also good, indicating that a substantial percentage of its positive predictions were correct.

*   **Decision Tree:** The Decision Tree model performed reasonably well, but was slightly outperformed by the Random Forest.

*   **Gradient Boosting and Logistic Regression:**  These models had lower performance across all metrics compared to the Random Forest and Decision Tree.  Their lower recall scores suggest they were less effective at identifying individuals who reported depression.

*   **Why Random Forest Might Be Best:** Random Forests are ensemble methods that combine multiple decision trees. This often leads to more robust and generalizable models compared to single decision trees or simpler models like logistic regression. They can handle complex interactions between features and are less prone to overfitting.

In conclusion, the Random Forest model demonstrates the best predictive performance for identifying individuals with a history of stroke who are likely to report a diagnosis of depression, based on the BRFSS data. The high AUC-ROC value suggests good discriminatory power, and the balanced F1-score indicates a good trade-off between precision and recall.
