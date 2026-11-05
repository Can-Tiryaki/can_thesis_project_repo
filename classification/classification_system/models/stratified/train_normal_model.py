import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from prepare_data_cls import get_classification_data

def run_tabakali_normal():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "saved_models")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "="*90)
    print("🔹 [1/5] TABAKALI BÖLME - VARSAYILAN PARALEL MODEL EĞİTİMLERİ")
    print("="*90)
    
    X, y, y_reg, class_names = get_classification_data()
    n_classes = len(class_names)
    labels = list(class_names.keys())
    target_names = [class_names[i] for i in labels]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = {
        'Random_Forest': RandomForestClassifier(random_state=42),
        'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
        'Extra_Trees': ExtraTreesClassifier(random_state=42),
        'Hist_Gradient_Boosting': HistGradientBoostingClassifier(random_state=42),
        'Artificial_Neural_Network': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(random_state=42, max_iter=1000))
        ]),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Decision_Tree': DecisionTreeClassifier(random_state=42)
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='mlogloss')

    all_results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        joblib.dump(model, os.path.join(models_dir, f"{name}_normal.pkl"))
        
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        test_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
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
            'Deney_Tipi': 'Tabakali_Normal',
            'Model': name,
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
        print(f"   -> {name:28s} | Test Acc: {test_acc:.4f} | Test F1 (Macro): {test_f1_macro:.4f} | ROC-AUC: {test_roc_auc:.4f}")

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(results_dir, "tabakali_normal_results.csv"), index=False)
    print(f"[✓] Tabakalı Normal sonuçları kaydedildi.")
    return df_results

if __name__ == "__main__":
    run_tabakali_normal()
