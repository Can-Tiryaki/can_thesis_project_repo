import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from prepare_data_cls import get_classification_data

def run_tabakali_tuned():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "saved_models")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "="*90)
    print("🛠️ [3/5] TABAKALI BÖLME - HİPERPARAMETRE OPTİMİZASYONLU (TUNED) EĞİTİMLER")
    print("="*90)
    
    X, y, y_reg, class_names = get_classification_data()
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Grid Search Parametre Alanları (Overfitting önleyici kısıtlar)
    param_grids = {
        'Random_Forest': (
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8, 12],
                'min_samples_split': [3, 5, 10],
                'min_samples_leaf': [2, 4]
            }
        ),
        'Gradient_Boosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'n_estimators': [50, 100],
                'learning_rate': [0.03, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'min_samples_split': [4, 8]
            }
        ),
        'Extra_Trees': (
            ExtraTreesClassifier(random_state=42, class_weight='balanced'),
            {
                'n_estimators': [50, 100, 150],
                'max_depth': [4, 6, 8],
                'min_samples_split': [3, 5, 10]
            }
        ),
        'AdaBoost': (
            AdaBoostClassifier(random_state=42),
            {
                'n_estimators': [30, 50, 100],
                'learning_rate': [0.05, 0.1, 0.5, 1.0]
            }
        ),
        'Decision_Tree': (
            DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [4, 8, 15],
                'ccp_alpha': [0.0, 0.01, 0.02]
            }
        )
    }

    all_results = []

    for name, (base_model, param_grid) in param_grids.items():
        print(f"--> Hiperparametre Araması Yapılıyor: {name}...")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        joblib.dump(best_model, os.path.join(models_dir, f"{name}_tuned.pkl"))
        
        train_preds = best_model.predict(X_train)
        test_preds = best_model.predict(X_test)
        test_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
        
        train_acc = accuracy_score(y_train, train_preds)
        train_f1_macro = f1_score(y_train, train_preds, average='macro')
        
        test_acc = accuracy_score(y_test, test_preds)
        test_prec_macro = precision_score(y_test, test_preds, average='macro', zero_division=0)
        test_rec_macro = recall_score(y_test, test_preds, average='macro', zero_division=0)
        test_f1_macro = f1_score(y_test, test_preds, average='macro', zero_division=0)
        test_f1_weighted = f1_score(y_test, test_preds, average='weighted', zero_division=0)
        
        test_roc_auc = roc_auc_score(y_test, test_proba, multi_class='ovr', average='macro') if test_proba is not None else np.nan

        acc_gap = train_acc - test_acc
        
        if acc_gap > 0.15:
            status = "⚠️ Yüksek Overfitting"
        elif acc_gap > 0.08:
            status = "🟡 Hafif Overfitting"
        else:
            status = "🟢 Dengeli Model"

        res = {
            'Deney_Tipi': 'Tabakali_Tuned',
            'Model': name,
            'Best_Params': str(grid.best_params_),
            'Train_Accuracy': train_acc,
            'Test_Accuracy': test_acc,
            'Accuracy_Gap': acc_gap,
            'Train_F1_Macro': train_f1_macro,
            'Test_F1_Macro': test_f1_macro,
            'Test_Precision_Macro': test_prec_macro,
            'Test_Recall_Macro': test_rec_macro,
            'Test_F1_Weighted': test_f1_weighted,
            'Test_ROC_AUC_OvR': test_roc_auc,
            'Durum': status
        }
        all_results.append(res)
        print(f"   -> {name:28s} | Test Acc: {test_acc:.4f} | Test F1 (Macro): {test_f1_macro:.4f} | Parametreler: {grid.best_params_}")

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(results_dir, "tabakali_tuned_results.csv"), index=False)
    print(f"[✓] Tabakalı Tuned sonuçları kaydedildi.")
    return df_results

if __name__ == "__main__":
    run_tabakali_tuned()
