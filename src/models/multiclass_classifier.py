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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from src.utils.logger import get_logger
from src.utils.config import MULTIFRACTAL_FEATURES_CSV, MULTICLASS_MODEL_DIR

logger = get_logger(__name__)

def perform_undersampling(df, target_col='category', strategy='majority'):
    """
    Undersamples the majority class ('healthy') to improve balance.
    Strategy:
    - 'majority': Matches 'healthy' count to the second largest class.
    - 'balanced': Matches all classes to the minority class (not recommended if minority is too small).
    """
    logger.info("Performing undersampling...")
    
    # Class counts
    counts = df[target_col].value_counts()
    logger.info(f"Original Counts:\n{counts}")
    
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    second_largest_count = counts.sort_values(ascending=False).iloc[1]

    # Split by class
    df_majority = df[df[target_col] == majority_class]
    df_rest = df[df[target_col] != majority_class]
    
    # Undersample majority
    df_majority_downsampled = resample(
        df_majority, 
        replace=False,    # sample without replacement
        n_samples=second_largest_count, # match second largest class
        random_state=42
    )
    
    # Combine back
    df_resampled = pd.concat([df_majority_downsampled, df_rest])
    
    logger.info(f"Resampled Counts:\n{df_resampled[target_col].value_counts()}")
    return df_resampled

def plot_confusion_matrix(y_true, y_pred, title, filename):
    MULTICLASS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    output_path = MULTICLASS_MODEL_DIR / filename
    plt.savefig(output_path)
    plt.close()

def evaluate_models(X, y, pca_dim=None):
    """
    Runs evaluation for multiple models on the given data.
    """
    desc = f"PCA={pca_dim}" if pca_dim else "No PCA"
    print(f"\n{'#'*20} Evaluation Context: {desc} {'#'*20}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Base Pipeline Steps
    steps_base = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    
    if pca_dim:
        steps_base.append(('pca', PCA(n_components=pca_dim)))

    # Models to Test
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, solver='saga'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps_base + [('clf', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = acc
        
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        # Only print full report if accuracy is decent or for specific models
        # print(classification_report(y_test, y_pred, zero_division=0))
        
        # Save CM for best performers or all? Let's save all for now specifically named
        plot_name = f"cm_{desc.replace(' ', '').replace('=', '')}_{name.replace(' ', '_')}.png"
        plot_confusion_matrix(y_test, y_pred, f"{name} ({desc})", plot_name)

    return results

def main():
    if not MULTIFRACTAL_FEATURES_CSV.exists():
        logger.error(f"Features file not found: {MULTIFRACTAL_FEATURES_CSV}")
        return

    # 1. Load Data
    df = pd.read_csv(MULTIFRACTAL_FEATURES_CSV)
    
    # 2. Undersampling
    # Remove 'healthy' data points to balance with 'neurological' (approx 278)
    # The user asked to "remove some healthy... since class imbalance"
    df_balanced = perform_undersampling(df, target_col='category')
    
    # Prepare X, y
    target_col = "category"
    drop_cols = ["filename", "category", "pathology", "speaker_id"]
    existing_drop_cols = [c for c in drop_cols if c in df_balanced.columns]
    
    X = df_balanced.drop(columns=existing_drop_cols)
    y = df_balanced[target_col]
    
    logger.info(f"balanced data shape: {X.shape}")

    # 3. PCA & Model Loop
    # Try different PCA dimensions
    # Total features ~45. 
    pca_dims = [None, 10, 20, 30] 
    
    all_results = []

    for dim in pca_dims:
        res = evaluate_models(X, y, pca_dim=dim)
        for model_name, acc in res.items():
            all_results.append({
                'PCA': str(dim) if dim else "All Features",
                'Model': model_name,
                'Accuracy': acc
            })

    # Summary
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values(by='Accuracy', ascending=False)
    
    print("\n" + "="*50)
    print("FINAL LEADERBOARD (Balanced Data)")
    print("="*50)
    print(summary_df.to_string(index=False))
    
    best_config = summary_df.iloc[0]
    print(f"\n🏆 Top Performer: {best_config['Model']} with {best_config['PCA']} (Acc: {best_config['Accuracy']:.4f})")

if __name__ == "__main__":
    main()
