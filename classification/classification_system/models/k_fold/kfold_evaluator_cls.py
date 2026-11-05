import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from prepare_data_cls import get_classification_data

def run_kfold_experiments(n_splits=5, is_weighted=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    mode_str = "Weighted" if is_weighted else "Normal"
    print("\n" + "="*90)
    print(f"🔄 [{4 if not is_weighted else 5}/5] {n_splits}-FOLD STRATIFIED CROSS VALIDATION ({mode_str.upper()})")
    print("="*90)

    X, y, y_reg, class_names = get_classification_data()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def get_models():
        if is_weighted:
            return {
                'Random_Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
                'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
                'Extra_Trees': ExtraTreesClassifier(class_weight='balanced', random_state=42),
                'Hist_Gradient_Boosting': HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
                'Artificial_Neural_Network': Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(random_state=42, max_iter=1000))
                ]),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Decision_Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
            }
        else:
            return {
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

    model_names = list(get_models().keys())
    results_list = []

    for name in model_names:
        fold_accs = []
        fold_f1_macros = []
        fold_prec_macros = []
        fold_rec_macros = []
        fold_rocs = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = get_models()[name]

            if is_weighted:
                sample_weights = compute_sample_weight('balanced', y_tr)
                if name in ['Gradient_Boosting', 'AdaBoost']:
                    model.fit(X_tr, y_tr, sample_weight=sample_weights)
                else:
                    model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr)

            val_preds = model.predict(X_val)
            val_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

            fold_accs.append(accuracy_score(y_val, val_preds))
            fold_f1_macros.append(f1_score(y_val, val_preds, average='macro', zero_division=0))
            fold_prec_macros.append(precision_score(y_val, val_preds, average='macro', zero_division=0))
            fold_rec_macros.append(recall_score(y_val, val_preds, average='macro', zero_division=0))
            
            if val_proba is not None:
                try:
                    fold_rocs.append(roc_auc_score(y_val, val_proba, multi_class='ovr', average='macro'))
                except Exception:
                    pass

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_f1 = np.mean(fold_f1_macros)
        std_f1 = np.std(fold_f1_macros)
        mean_prec = np.mean(fold_prec_macros)
        mean_rec = np.mean(fold_rec_macros)
        mean_roc = np.mean(fold_rocs) if len(fold_rocs) > 0 else np.nan

        res = {
            'Deney_Tipi': f'KFold_{n_splits}_{mode_str}',
            'Model': name,
            'Mean_CV_Accuracy': mean_acc,
            'Std_CV_Accuracy': std_acc,
            'Mean_CV_F1_Macro': mean_f1,
            'Std_CV_F1_Macro': std_f1,
            'Mean_CV_Precision_Macro': mean_prec,
            'Mean_CV_Recall_Macro': mean_rec,
            'Mean_CV_ROC_AUC': mean_roc
        }
        results_list.append(res)
        print(f"   -> {name:28s} | 5-Fold Acc: {mean_acc:.4f} (±{std_acc:.3f}) | F1 (Macro): {mean_f1:.4f} (±{std_f1:.3f}) | ROC-AUC: {mean_roc:.4f}")

    df_results = pd.DataFrame(results_list)
    out_file = os.path.join(results_dir, f"kfold_{n_splits}_{mode_str.lower()}_results.csv")
    df_results.to_csv(out_file, index=False)
    print(f"[✓] {n_splits}-Fold {mode_str} sonuçları kaydedildi: {out_file}")
    return df_results

if __name__ == "__main__":
    run_kfold_experiments(5, is_weighted=False)
    run_kfold_experiments(5, is_weighted=True)
