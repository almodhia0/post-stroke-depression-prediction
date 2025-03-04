import pandas as pd
from sklearn.impute import LogisticRegression
# Add other necessary imports (e.g., for feature engineering)

def preprocess_brfss_data(brfss_data_path):
    """
    Preprocesses the BRFSS data.

    Args:
        brfss_data_path: Path to the CSV file containing the BRFSS data.

    Returns:
        A pandas DataFrame containing the preprocessed data.
    """
    # 1. Load Data
    df = pd.read_csv(brfss_data_path)

    # 2. Feature Selection (Select the 14 columns you used)
    selected_columns = [
        "gender", "age_group", "marital_status", "education_level",
        "income_category", "bmi_category", "physical_activity",
        "smoking_status", "diabetes_status", "hypertension_status",
        "cholesterol_status", "stroke_status", "depressive_disorder",
        "alcohol_status"
    ]
    df = df[selected_columns]


    # 3. Create depression_stroke variable
    df['depression_stroke'] = 0
    df.loc[(df['stroke_status'] == 1) & (df['depressive_disorder'] == 1), 'depression_stroke'] = 1

    #4. Drop rows with missing values in 'stroke_status' or 'depressive_disorder'
    df.dropna(subset=['stroke_status', 'depressive_disorder'], inplace=True)


    # 5. Impute Missing Values (using Logistic Regression - example for one variable)
    #    Repeat this for each variable with missing values that needs imputation.
    for col in df.columns:
      if df[col].isnull().any():
        # Prepare data for imputation
        df_missing = df[df[col].isnull()]
        df_not_missing = df[df[col].notnull()]

        X_train = df_not_missing.drop(columns=[col, 'depression_stroke'])
        y_train = df_not_missing[col]
        X_test = df_missing.drop(columns=[col,  'depression_stroke'])
        # Handle cases with too few samples for some classes
        if len(y_train.unique()) < 2:  # Need at least 2 classes for LogisticRegression
            # Simplest approach: fill with the most frequent value
              most_frequent = df_not_missing[col].mode()[0]
              df.loc[df[col].isnull(), col] = most_frequent
              print(f"Warning: Imputed '{col}' using most frequent value due to insufficient class diversity.")
              continue
        #Train Model
        imputer = LogisticRegression(solver='liblinear', random_state=42)
        try:
                imputer.fit(X_train, y_train)
                # Predict and impute missing values
                imputed_values = imputer.predict(X_test)
                df.loc[df[col].isnull(), col] = imputed_values
        except ValueError as e:
                print(f"Error imputing '{col}': {e}.  Filling with most frequent value instead.")
                most_frequent_value = df[col].mode()[0]  # Get the most frequent value
                df[col] = df[col].fillna(most_frequent_value)  # Fill NaN with the most frequent
    # 6. Return Preprocessed DataFrame
    return df

if __name__ == '__main__':
    # Example Usage (Replace with your actual file path)
    preprocessed_df = preprocess_brfss_data("path/to/your/brfss_data.csv")
    print(preprocessed_df.head())
    # Save the preprocessed data (optional)
    preprocessed_df.to_csv("data/preprocessed_brfss_data.csv", index=False)
