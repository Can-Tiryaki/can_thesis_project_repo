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

# Sınıflandırma Model Tanımları
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

# Veri yükleyici
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_data_cls import get_classification_data

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "saved_models")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("==========================================================================================")
    print("🚀 HbA1c TIBBİ RİSK SINIFLANDIRMA DENEYLERİ BAŞLIYOR (ADA KILAVUZU EŞİKLERİ)")
    print("==========================================================================================")
    
    # 1. Veri Yükleme ve Sınıflandırma
    X, y, y_reg, class_names = get_classification_data()
    n_classes = len(class_names)
    labels = list(class_names.keys())
    target_names = [class_names[i] for i in labels]
    
    print(f"\n[*] Toplam Hasta Sayısı: {len(X)}")
    print("[*] Sınıf Dağılımı:")
    unique, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(unique, counts):
        pct = (cnt / len(y)) * 100
        print(f"   - {class_names[c]}: {cnt} hasta (%{pct:.2f})")

    # 1.1 Sınıf Dağılımı Grafiği Çiz ve Kaydet
    plt.figure(figsize=(8, 5))
    bars = plt.bar(target_names, counts, color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    plt.title('HbA1c Klinik Risk Sınıfı Dağılımı (ADA Eşikleri)', fontsize=13, fontweight='bold')
    plt.xlabel('HbA1c Sınıfları', fontsize=11, fontweight='bold')
    plt.ylabel('Hasta Sayısı', fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(y)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                 f'{height}\n(%{pct:.1f})', ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    dist_chart_path = os.path.join(results_dir, "class_distribution.png")
    plt.savefig(dist_chart_path, dpi=150)
    plt.close()
    print(f"[✓] Sınıf dağılım grafiği kaydedildi: {dist_chart_path}")

    # 2. Tabakalı Bölme (Stratified Split %80 Train, %20 Test)
    # Regresyondaki aynı random_state=42 ile veri bölünmesi
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\n[*] Bölme Tamamlandı (Stratified 80/20). Train: {len(X_train)} hasta, Test: {len(X_test)} hasta.")

    # 3. Sınıflandırma Modelleri
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
    test_predictions = {}
    test_probabilities = {}

    print("\n[*] Sınıflandırma Modelleri Eğitiliyor ve Değerlendiriliyor...")

    for name, model in models.items():
        print(f"--> Model Eğitiliyor: {name}...")
        
        # Eğit
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Modeli kaydet
        model_file = os.path.join(models_dir, f"{name}_cls_model.pkl")
        joblib.dump(model, model_file)
        
        # Tahminler
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        test_predictions[name] = test_preds
        
        # Olasılıklar (ROC-AUC için)
        if hasattr(model, "predict_proba"):
            test_proba = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            test_proba = model.decision_function(X_test)
        else:
            test_proba = None
        test_probabilities[name] = test_proba
        
        # Train Metrikleri
        train_acc = accuracy_score(y_train, train_preds)
        train_f1_macro = f1_score(y_train, train_preds, average='macro')
        
        # Test Metrikleri
        test_acc = accuracy_score(y_test, test_preds)
        test_prec_macro = precision_score(y_test, test_preds, average='macro', zero_division=0)
        test_rec_macro = recall_score(y_test, test_preds, average='macro', zero_division=0)
        test_f1_macro = f1_score(y_test, test_preds, average='macro', zero_division=0)
        
        test_prec_weighted = precision_score(y_test, test_preds, average='weighted', zero_division=0)
        test_rec_weighted = recall_score(y_test, test_preds, average='weighted', zero_division=0)
        test_f1_weighted = f1_score(y_test, test_preds, average='weighted', zero_division=0)
        
        # ROC-AUC (Multiclass OvR)
        if test_proba is not None:
            try:
                test_roc_auc = roc_auc_score(y_test, test_proba, multi_class='ovr', average='macro')
            except Exception as e:
                test_roc_auc = np.nan
        else:
            test_roc_auc = np.nan

        acc_gap = train_acc - test_acc
        f1_gap = train_f1_macro - test_f1_macro
        
        if acc_gap > 0.15:
            status = "⚠️ Yüksek Overfitting"
        elif acc_gap > 0.08:
            status = "🟡 Hafif Overfitting"
        elif acc_gap < 0.02 and test_acc < 0.60:
            status = "🔴 Underfitting"
        else:
            status = "🟢 Dengeli Model"

        res = {
            'Model': name,
            'Train_Accuracy': train_acc,
            'Test_Accuracy': test_acc,
            'Accuracy_Gap': acc_gap,
            'Train_F1_Macro': train_f1_macro,
            'Test_F1_Macro': test_f1_macro,
            'Test_Precision_Macro': test_prec_macro,
            'Test_Recall_Macro': test_rec_macro,
            'Test_F1_Weighted': test_f1_weighted,
            'Test_Precision_Weighted': test_prec_weighted,
            'Test_Recall_Weighted': test_rec_weighted,
            'Test_ROC_AUC_OvR': test_roc_auc,
            'Durum': status
        }
        all_results.append(res)
        print(f"   Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test F1 (Macro): {test_f1_macro:.4f} | ROC-AUC: {test_roc_auc:.4f}")

    # DataFrame oluştur
    df_results = pd.DataFrame(all_results)
    
    # CSV kaydet
    csv_path = os.path.join(results_dir, "classification_comparison_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[✓] Sayısal sonuçlar CSV olarak kaydedildi: {csv_path}")

    # En başarılı test modelini bul (Test_F1_Macro veya Test_Accuracy bazlı)
    best_idx = df_results['Test_F1_Macro'].idxmax()
    best_model_name = df_results.loc[best_idx, 'Model']
    best_fitted_model = trained_models[best_model_name]
    print(f"\n🏆 En Başarılı Sınıflandırma Modeli: {best_model_name} (Test F1 Macro: {df_results.loc[best_idx, 'Test_F1_Macro']:.4f}, Test Acc: {df_results.loc[best_idx, 'Test_Accuracy']:.4f})")

    # 4. Yan Yana Karşılaştırma Bar Grafiği (Train vs Test Accuracy & F1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x_indices = np.arange(len(df_results))
    width = 0.35
    
    # 4.1 Accuracy Grafiği
    axes[0].bar(x_indices - width/2, df_results['Train_Accuracy'], width, label='Eğitim (Train) Accuracy', color='#3498db', edgecolor='black')
    axes[0].bar(x_indices + width/2, df_results['Test_Accuracy'], width, label='Test (Test) Accuracy', color='#e74c3c', edgecolor='black')
    axes[0].set_xlabel('Modeller', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy (Doğruluk)', fontsize=11, fontweight='bold')
    axes[0].set_title('Sınıflandırma Modelleri Accuracy Karşılaştırması', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(df_results['Model'], rotation=20, fontsize=9)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=10)
    
    for idx, row in df_results.iterrows():
        axes[0].text(idx - width/2, row['Train_Accuracy'] + 0.01, f"{row['Train_Accuracy']:.2f}", ha='center', va='bottom', fontsize=8)
        axes[0].text(idx + width/2, row['Test_Accuracy'] + 0.01, f"{row['Test_Accuracy']:.2f}", ha='center', va='bottom', fontsize=8)

    # 4.2 F1-Score Macro Grafiği
    axes[1].bar(x_indices - width/2, df_results['Train_F1_Macro'], width, label='Eğitim (Train) F1 Macro', color='#2ecc71', edgecolor='black')
    axes[1].bar(x_indices + width/2, df_results['Test_F1_Macro'], width, label='Test (Test) F1 Macro', color='#9b59b6', edgecolor='black')
    axes[1].set_xlabel('Modeller', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('F1-Score (Macro)', fontsize=11, fontweight='bold')
    axes[1].set_title('Sınıflandırma Modelleri F1-Macro Skoru Karşılaştırması', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(df_results['Model'], rotation=20, fontsize=9)
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend(fontsize=10)
    
    for idx, row in df_results.iterrows():
        axes[1].text(idx - width/2, row['Train_F1_Macro'] + 0.01, f"{row['Train_F1_Macro']:.2f}", ha='center', va='bottom', fontsize=8)
        axes[1].text(idx + width/2, row['Test_F1_Macro'] + 0.01, f"{row['Test_F1_Macro']:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    acc_f1_chart_path = os.path.join(results_dir, "train_test_accuracy_f1_comparison.png")
    plt.savefig(acc_f1_chart_path, dpi=150)
    plt.close()
    print(f"[✓] Accuracy ve F1 Karşılaştırma grafiği kaydedildi: {acc_f1_chart_path}")

    # 5. Tüm Modeller İçin Confusion Matrix Görsellerini Çiz
    num_models = len(test_predictions)
    n_cols = 4 if num_models >= 7 else 3
    n_rows = (num_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten() if num_models > 1 else [axes]
    
    for idx, (name, test_preds) in enumerate(test_predictions.items()):
        cm = confusion_matrix(y_test, test_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=target_names, yticklabels=target_names,
                    cbar=False, annot_kws={"size": 11, "weight": "bold"})
        
        test_f1 = df_results.loc[df_results['Model'] == name, 'Test_F1_Macro'].values[0]
        test_acc = df_results.loc[df_results['Model'] == name, 'Test_Accuracy'].values[0]
        
        axes[idx].set_title(f"{name}\nAcc: {test_acc:.4f} | F1-Macro: {test_f1:.4f}", fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Tahmin Edilen Sınıf', fontsize=9)
        axes[idx].set_ylabel('Gerçek Sınıf', fontsize=9)

    # Hide unused subplots
    for i in range(num_models, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    cm_chart_path = os.path.join(results_dir, "all_models_confusion_matrices.png")
    plt.savefig(cm_chart_path, dpi=150)
    plt.close()
    print(f"[✓] Tüm modellerin Karmaşıklık Matrisi (Confusion Matrix) grafiği kaydedildi: {cm_chart_path}")

    # 6. En Başarılı Model İçin Çoklu Sınıf ROC-AUC Eğrisi Çiz
    best_proba = test_probabilities[best_model_name]
    if best_proba is not None:
        y_test_bin = label_binarize(y_test, classes=labels)
        from sklearn.metrics import roc_curve, auc

        plt.figure(figsize=(8, 6))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], best_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'ROC {class_names[i]} (AUC = {roc_auc_val:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Rastgele Tahmin (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)', fontsize=11, fontweight='bold')
        plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)', fontsize=11, fontweight='bold')
        plt.title(f'En Başarılı Model ({best_model_name}) ROC Eğrileri (OvR)', fontsize=13, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        roc_chart_path = os.path.join(results_dir, "best_model_roc_curve.png")
        plt.savefig(roc_chart_path, dpi=150)
        plt.close()
        print(f"[✓] En başarılı model ROC Eğrisi grafiği kaydedildi: {roc_chart_path}")

    # 7. En Başarılı Model Özellik Önem Sırası (Feature Importance)
    if hasattr(best_fitted_model, 'feature_importances_') or (hasattr(best_fitted_model, 'steps') and hasattr(best_fitted_model.steps[-1][1], 'feature_importances_')):
        if hasattr(best_fitted_model, 'feature_importances_'):
            importances = best_fitted_model.feature_importances_
        else:
            importances = best_fitted_model.steps[-1][1].feature_importances_
            
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=X.columns[indices], hue=X.columns[indices], palette='magma', legend=False)
        plt.xlabel('Özellik Skoru (Gini Importance)', fontsize=11, fontweight='bold')
        plt.ylabel('Girdi Değişkenleri', fontsize=11, fontweight='bold')
        plt.title(f'En Başarılı Sınıflandırma Modeli ({best_model_name}) Özellik Önem Sırası', fontsize=13, fontweight='bold')
        plt.tight_layout()
        importance_path = os.path.join(results_dir, "best_model_feature_importance.png")
        plt.savefig(importance_path, dpi=150)
        plt.close()
        print(f"[✓] Sınıflandırma özellik önem sırası grafiği kaydedildi: {importance_path}")

    # 8. Markdown Raporu Oluştur
    markdown_lines = []
    markdown_lines.append("# 📊 HbA1c Klinik Risk Sınıflandırması Model Sonuçları (ADA Eşikleri)")
    markdown_lines.append("Bu rapor, temizlenmiş hasta verilerinin (%80 Eğitim, %20 Test) olarak tabakalı (Stratified Split) şekilde bölünmesi ve klinik HbA1c risk sınıflarının (**0: Normal**, **1: Prediyabet**, **2: Diyabet**) sınıflandırma algoritmalarıyla eğitilmesi sonucu elde edilen metrikleri içermektedir.\n")
    
    markdown_lines.append("## 📈 Sınıf Dağılımı")
    for c, cnt in zip(unique, counts):
        pct = (cnt / len(y)) * 100
        markdown_lines.append(f"- **Sınıf {c} ({class_names[c]})**: {cnt} hasta (%{pct:.2f})")
    markdown_lines.append("\n")

    markdown_lines.append("## 📋 Karşılaştırmalı Model Başarı Tablosu\n")
    markdown_lines.append("| Model | Train Acc | Test Acc | Acc Gap | Train F1 (Macro) | Test F1 (Macro) | Precision (Macro) | Recall (Macro) | F1 (Weighted) | ROC-AUC (OvR) | Durum |")
    markdown_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for idx, row in df_results.iterrows():
        is_best = (idx == best_idx)
        
        def fmt(val):
            if pd.isna(val):
                return "N/A"
            txt = f"{val:.4f}"
            return f"**{txt}**" if is_best else txt

        model_name_fmt = f"**{row['Model']}**" if is_best else row['Model']
        status_fmt = f"**{row['Durum']}**" if is_best else row['Durum']
        
        line = f"| {model_name_fmt} | {fmt(row['Train_Accuracy'])} | {fmt(row['Test_Accuracy'])} | {fmt(row['Accuracy_Gap'])} | {fmt(row['Train_F1_Macro'])} | {fmt(row['Test_F1_Macro'])} | {fmt(row['Test_Precision_Macro'])} | {fmt(row['Test_Recall_Macro'])} | {fmt(row['Test_F1_Weighted'])} | {fmt(row['Test_ROC_AUC_OvR'])} | {status_fmt} |"
        markdown_lines.append(line)

    # Detaylı En İyi Model Sınıflandırma Raporu
    best_preds = test_predictions[best_model_name]
    cls_rep = classification_report(y_test, best_preds, target_names=target_names)
    
    markdown_lines.append("\n## 🏆 En Başarılı Model Detaylı Sınıf Raporu: " + best_model_name + "\n")
    markdown_lines.append("```text\n" + cls_rep + "\n```")

    markdown_content = "\n".join(markdown_lines)
    
    md_file = os.path.join(results_dir, "classification_comparison_results.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"[✓] Markdown raporu kaydedildi: {md_file}")

    print("\n" + "="*110)
    print(f"📋 HBA1C SINIFLANDIRMA MODEL SONUÇLARI OZETI (EN İYİ MODEL: {best_model_name})")
    print("="*110)
    print(markdown_content)
    print("="*110)

if __name__ == "__main__":
    main()
