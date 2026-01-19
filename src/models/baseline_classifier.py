import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger
from src.utils.config import CLASSIC_FEATURES_CSV, BASELINE_MODEL_DIR, REPORTS_DIR

logger = get_logger(__name__)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    output_path = BASELINE_MODEL_DIR / filename
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")

def train_and_evaluate(name, pipeline, param_grid, X_train, y_train, X_test, y_test):
    print(f"\n{'='*20} {name} {'='*20}")
    
    # 1. Default Performance
    logger.info(f"Training default {name}...")
    pipeline.fit(X_train, y_train)
    y_pred_default = pipeline.predict(X_test)
    acc_default = accuracy_score(y_test, y_pred_default)
    print(f"Default Accuracy: {acc_default:.4f}")
    print(classification_report(y_test, y_pred_default, zero_division=0))
    
    # 2. Hyperparameter Tuning
    logger.info(f"Tuning {name} hyperparameters...")
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    acc_tuned = accuracy_score(y_test, y_pred_tuned)
    
    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"Tuned Accuracy: {acc_tuned:.4f} (Improvement: {acc_tuned - acc_default:+.4f})")
    print(classification_report(y_test, y_pred_tuned, zero_division=0))
    
    plot_confusion_matrix(
        y_test, 
        y_pred_tuned, 
        f"{name} Confusion Matrix (Tuned)", 
        f"cm_{name.lower().replace(' ', '_')}.png"
    )
    
    return acc_tuned, best_model

def main():
    if not CLASSIC_FEATURES_CSV.exists():
        logger.error(f"Data file not found: {CLASSIC_FEATURES_CSV}")
        return

    # Load Data
    logger.info("Loading data...")
    df = pd.read_csv(CLASSIC_FEATURES_CSV)
    
    # Separate X and y
    target_col = "category"
    drop_cols = ["filename", "category", "pathology", "speaker_id"]
    
    # Verify columns exist before dropping
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=existing_drop_cols)
    y = df[target_col]
    
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logger.info(f"Class distribution:\n{y.value_counts()}")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Models
    
    # 1. Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000))
    ])
    
    lr_params = {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__solver': ['lbfgs', 'saga']
    }
    
    # 2. Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    rf_params = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }

    # 3. SVM (RBF Kernel) Pipeline
    svm_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    ])
    
    svm_params = {
        'clf__C': [0.1, 1.0, 10.0, 100.0],
        'clf__gamma': ['scale', 'auto', 0.01, 0.1]
    }

    # Execution
    results = {}
    
    results['Logistic Regression'], _ = train_and_evaluate(
        "Logistic Regression", lr_pipeline, lr_params, X_train, y_train, X_test, y_test
    )
    
    results['Random Forest'], _ = train_and_evaluate(
        "Random Forest", rf_pipeline, rf_params, X_train, y_train, X_test, y_test
    )
    
    results['SVM (RBF)'], _ = train_and_evaluate(
        "SVM (RBF)", svm_pipeline, svm_params, X_train, y_train, X_test, y_test
    )
    
    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY")
    print("="*40)
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")
        
    best_model_name = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best_model_name}")

if __name__ == "__main__":
    main()
