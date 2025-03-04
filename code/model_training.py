import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np #Import numpy

def train_and_evaluate_models(data_path):
    """Trains, evaluates, and validates ML models for PSD prediction."""

    # 1. Load Data
    df = pd.read_csv(data_path)

    # 2. Prepare Data
    X = df.drop(columns=['depression_stroke'])
    y = df['depression_stroke']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Handle Class Imbalance (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 5. Define Models and Hyperparameter Grids
    models = {
        'RandomForest': (RandomForestClassifier(random_state=42),
                         {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}),
        'DecisionTree': (DecisionTreeClassifier(random_state=42),
                         {'max_depth': [3, 5, 7, 10]}),
        'GradientBoosting': (GradientBoostingClassifier(random_state=42),
                             {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}),
        'LogisticRegression': (LogisticRegression(random_state=42, solver='liblinear'),
                               {'C': [0.1, 1, 10]})
    }

    # 6. Train, Tune, and Evaluate Models
    results = {}
    best_model = None
    best_score = 0

    for model_name, (model, param_grid) in models.items():
        print(f"Training and evaluating {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')  # Use F1-score for optimization
        grid_search.fit(X_train_resampled, y_train_resampled)

        best_model_instance = grid_search.best_estimator_

        # Evaluate on Test Set
        y_pred = best_model_instance.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, best_model_instance.predict_proba(X_test)[:, 1])

        results[model_name] = {
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

        # Track Best Model (using F1-score)
        if f1 > best_score:
            best_score = f1
            best_model = (model_name, best_model_instance)
    # 7. Feature Importance (example with best model - adapt as needed)
    if best_model:
        print(f"\nBest Model: {best_model[0]}")
        print(f"Best Parameters: {results[best_model[0]]['best_params']}")

        if hasattr(best_model[1], 'feature_importances_'): #For tree-based
            importances = best_model[1].feature_importances_
        elif hasattr(best_model[1], 'coef_'): # For linear models
            importances = best_model[1].coef_[0] # Access the coefficients
        else:
           importances = None # No feature importance attribute

        if importances is not None:
              feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
              feature_importances = feature_importances.sort_values('importance', ascending=False)
              print("\nFeature Importances:")
              print(feature_importances)
              feature_importances.to_csv('results/feature_importance.csv', index = False)

    # 8. Mutual Information
    mi_scores = mutual_info_classif(X_train_resampled, y_train_resampled)
    ami_scores = adjusted_mutual_info_score(y_train_resampled, mutual_info_classif(X_train_resampled, y_train_resampled, discrete_features=True))

    mi_df = pd.DataFrame({'Feature': X.columns, 'MI
