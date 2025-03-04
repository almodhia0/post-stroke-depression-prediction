# post-stroke-depression-prediction
# Prediction of Depression in Stroke Survivors using BRFSS Data

This repository contains the code and supplementary materials associated with the study "post-stroke-depression-prediction", which develops a machine learning model to predict the probability of reported depression among individuals with a history of stroke, using data from the Behavioral Risk Factor Surveillance System (BRFSS).

**Project Overview:**

This project utilizes the 2023 BRFSS dataset (released September 2024) to build a predictive model for reported depression in individuals who have experienced a stroke. The model uses various demographic and health-related factors (listed below) to estimate the probability of an individual reporting a diagnosis of depression.  It's important to note that the model predicts the probability of *reporting* a diagnosis, not the presence of active depression.  This aligns with the "ever diagnosed" questions used in the BRFSS.

**Data Source:**

The BRFSS dataset is publicly available and maintained by the U.S. Centers for Disease Control and Prevention (CDC).  It is released under the CC0 1.0 Universal Public Domain Dedication license.  You can access the data and documentation here:

https://www.cdc.gov/brfss/annual_data/annual_2023.html

The file `data/BRFSS_data_description.txt` provides further details on accessing and understanding the relevant variables within the BRFSS dataset.  **We do not include the full BRFSS dataset in this repository due to its size and the CDC's distribution policies.**

**Variables Used:**

The model utilizes the following variables from the BRFSS:

| Variable             | Definition                                                                                                                                                                        |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `gender`             | 0 - Female , 1 - Male                                                                                                                                                              |
| `age_group`          | 1- 18 to 24 , 2- 25 to 29 , 3- 30 to 34 , 4- 35 to 39 , 5- 40 to 44 , 6- 45 to 49 , 7- 50 to 54 , 8- 55 to 59 , 9- 60 to 64 , 10- 65 to 69 , 11- 70 to 74 , 12- 75 to 79 , 13- 80 or older |
| `marital_status`     | 0 - Not Married, 1 - Married                                                                                                                                                     |
| `education_level`    | 1 - Did not graduate High School , 2 - Graduated High School , 3 - Attended College or Technical School , 4 - Graduated from College or Technical School                            |
| `income_category`    | 1 - Less than $15,000 , 2 - $15,000 to < $25,000 , 3 - $25,000 to < $35,000 , 4 - $35,000 to < $50,000 , 5 - $50,000 to < $100,000 , 6 - $100,000 to < $200,000 , 7 - $200,000 or more  |
| `bmi_category`       | 1 - Underweight , 2 - Normal Weight , 3 - Overweight , 4 - Obese                                                                                                                |
| `physical_activity`  | 0 - No physical activity or exercise in last 30 days , 1 - Had physical activity or exercise                                                                                      |
| `smoking_status`     | 0 - No , 1 - Yes                                                                                                                                                                  |
| `diabetes_status`    | 0 - Not diabetic 1 - Diabetic                                                                                                                                                     |
| `hypertension_status`| 0 - No hypertension , 1 - Have hypertension                                                                                                                                        |
| `cholesterol_status` | 0 - Normal , 1 - High                                                                                                                                                             |
| `stroke_status`      | 0 - No, 1 - Yes                                                                                                                                                                  |
| `depressive_disorder`| 0 - No, 1 - Yes                                                                                                                                                                  |
| `alcohol_status`     | 0 - Do not drink , 1 - Yes                                                                                                                                                         |
|`Depression_stroke`| 0-No depression in stroke patient , 1 - Yes|

**Code Structure:**

*   `code/preprocessing.py`: This script contains the Python code used for data cleaning, imputation of missing values (using logistic regression), feature selection, and feature engineering, as described in the Methods section of the paper.
*   `code/model_training.py`: This script implements the model selection, training, evaluation, and validation procedures.  It includes the code for training the four machine learning models (Random Forest, Decision Tree, Gradient Boosting, and Logistic Regression), performing hyperparameter optimization using GridSearchCV, and evaluating performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.  SMOTE is used to address class imbalance.
* `code/requirements.txt`: This file lists the Python packages required to run the code.

**How to Use:**

1.  **Download the BRFSS Data:** Obtain the 2023 BRFSS data from the CDC website (link provided above).
2.  **Install Dependencies:**  Install the required Python packages using pip:
    ```bash
    pip install -r code/requirements.txt
    ```
3.  **Run the Scripts:** Execute the scripts in the `code` directory in the following order:
    *   `preprocessing.py` (to prepare the data)
    *   `model_training.py` (to train and evaluate the models)
    *   `model_simplified.py` (OPTIONAL - to use the simplified model)

**Results:**
The 'results' folder include the results of feature importance and model performance, for easy analysis.

**Limitations:**

The model predicts the probability of *reporting* a depression diagnosis, not necessarily the presence of *current* or *post-stroke* depression specifically.  The "ever diagnosed" nature of the BRFSS questions limits our ability to determine the precise temporal relationship between stroke and depression onset.

**Citation:**

If you use this code or the findings of our study, please cite:

Link will be provided soon....

**Contact:**

dr_alqahtani@hotmail.com
