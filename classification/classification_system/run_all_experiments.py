import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Dizin ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.tabakali.train_normal_model import run_tabakali_normal
from models.tabakali.train_weighted_model import run_tabakali_weighted
from models.tabakali.train_tuned_model import run_tabakali_tuned
from models.k_fold.kfold_evaluator_cls import run_kfold_experiments

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("==========================================================================================")
    print("🚀 KAPSAMLI HBA1C KLİNİK RİSK SINIFLANDIRMA DENEYLERİ SUITE BAŞLIYOR (ADA KILAVUZU)")
    print("==========================================================================================")

    # 1. Tabakalı Normal Deneyleri
    df_tabakali_normal = run_tabakali_normal()

    # 2. Tabakalı Weighted Deneyleri
    df_tabakali_weighted = run_tabakali_weighted()

    # 3. Tabakalı Tuned Deneyleri
    df_tabakali_tuned = run_tabakali_tuned()

    # 4. K-Fold Normal Deneyleri
    df_kfold_normal = run_kfold_experiments(n_splits=5, is_weighted=False)

    # 5. K-Fold Weighted Deneyleri
    df_kfold_weighted = run_kfold_experiments(n_splits=5, is_weighted=True)

    print("\n" + "="*90)
    print("📊 TÜM DENEY SONUÇLARI BİRLEŞTİRİLİYOR VE KARŞILAŞTIRMALI ANALİZ ÜRETİLİYOR...")
    print("="*90)

    # DataFrame'leri Düzenleme ve Birleştirme
    combined_rows = []

    # Tabakalılar
    for df in [df_tabakali_normal, df_tabakali_weighted, df_tabakali_tuned]:
        for idx, row in df.iterrows():
            combined_rows.append({
                'Deney_Tipi': row['Deney_Tipi'],
                'Model': row['Model'],
                'Test_Accuracy': row['Test_Accuracy'],
                'Test_F1_Macro': row['Test_F1_Macro'],
                'Test_Precision_Macro': row['Test_Precision_Macro'],
                'Test_Recall_Macro': row['Test_Recall_Macro'],
                'ROC_AUC': row['Test_ROC_AUC_OvR'],
                'Durum': row['Durum']
            })

    # K-Fold'lar
    for df in [df_kfold_normal, df_kfold_weighted]:
        for idx, row in df.iterrows():
            combined_rows.append({
                'Deney_Tipi': row['Deney_Tipi'],
                'Model': row['Model'],
                'Test_Accuracy': row['Mean_CV_Accuracy'],
                'Test_F1_Macro': row['Mean_CV_F1_Macro'],
                'Test_Precision_Macro': row['Mean_CV_Precision_Macro'],
                'Test_Recall_Macro': row['Mean_CV_Recall_Macro'],
                'ROC_AUC': row['Mean_CV_ROC_AUC'],
                'Durum': f"CV Std: ±{row['Std_CV_F1_Macro']:.3f}"
            })

    df_master = pd.DataFrame(combined_rows)
    master_csv = os.path.join(results_dir, "master_classification_experiments.csv")
    df_master.to_csv(master_csv, index=False)
    print(f"[✓] Tüm deneylerin birleşik CSV sonucu kaydedildi: {master_csv}")

    # En yüksek F1-Macro alan modeli bul
    best_idx = df_master['Test_F1_Macro'].idxmax()
    best_row = df_master.loc[best_idx]
    
    print("\n" + "🏆 "*15)
    print(f"EN BAŞARILI SINIFLANDIRMA DENEYİ VE MODELİ:")
    print(f"   Deney Tipi : {best_row['Deney_Tipi']}")
    print(f"   Model      : {best_row['Model']}")
    print(f"   F1 (Macro) : {best_row['Test_F1_Macro']:.4f}")
    print(f"   Accuracy   : {best_row['Test_Accuracy']:.4f}")
    print(f"   Recall     : {best_row['Test_Recall_Macro']:.4f}")
    print("🏆 "*15)

    # Karşılaştırma Grafiği: Deney Tiplerine Göre F1-Macro Skorları (Random Forest & Gradient Boosting vb.)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_master, x='Model', y='Test_F1_Macro', hue='Deney_Tipi', palette='tab10')
    plt.title('Farklı Deney Mimarilerinde Modellerin F1-Macro Skor Karşılaştırması', fontsize=14, fontweight='bold')
    plt.xlabel('Sınıflandırma Modelleri', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score (Macro)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=20)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Deney Tipi', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    master_chart_path = os.path.join(results_dir, "master_experiments_f1_comparison.png")
    plt.savefig(master_chart_path, dpi=150)
    plt.close()
    print(f"[✓] Kapsamlı deney karşılaştırma grafiği kaydedildi: {master_chart_path}")

    # MASTER MARKDOWN RAPORU
    md_lines = []
    md_lines.append("# 🏆 HbA1c Klinik Risk Sınıflandırması MASTER DENEY RAPORU (ADA Kılavuzu)")
    md_lines.append("Bu rapor; **Tabakalı Bölme**, **5-Fold Çapraz Doğrulama**, **Sınıf Ağırlıklandırmalı (Weighted)** ve **Hiperparametre Optimizasyonlu (Tuned)** tüm sınıflandırma deneylerinin özetini içermektedir.\n")

    md_lines.append(f"### 👑 En Yüksek F1-Macro Başarısı Gösteren Deney:")
    md_lines.append(f"- **Deney Tipi**: `{best_row['Deney_Tipi']}`")
    md_lines.append(f"- **Model**: `{best_row['Model']}`")
    md_lines.append(f"- **Test F1 (Macro)**: `{best_row['Test_F1_Macro']:.4f}`")
    md_lines.append(f"- **Test Accuracy**: `{best_row['Test_Accuracy']:.4f}`")
    md_lines.append(f"- **Test Recall (Macro)**: `{best_row['Test_Recall_Macro']:.4f}`\n")

    md_lines.append("## 📋 TÜM DENEYLERİN ŞAMPİYONLUK VE BAŞARI TABLOSU\n")
    md_lines.append("| Deney Tipi | Model | Test Accuracy | Test F1 (Macro) | Precision (Macro) | Recall (Macro) | ROC-AUC (OvR) | Durum / Not |")
    md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for idx, row in df_master.iterrows():
        is_best = (idx == best_idx)
        def fmt(val):
            if pd.isna(val): return "N/A"
            txt = f"{val:.4f}"
            return f"**{txt}**" if is_best else txt
        
        dt_fmt = f"**{row['Deney_Tipi']}**" if is_best else row['Deney_Tipi']
        m_fmt = f"**{row['Model']}**" if is_best else row['Model']
        
        line = f"| {dt_fmt} | {m_fmt} | {fmt(row['Test_Accuracy'])} | {fmt(row['Test_F1_Macro'])} | {fmt(row['Test_Precision_Macro'])} | {fmt(row['Test_Recall_Macro'])} | {fmt(row['ROC_AUC'])} | {row['Durum']} |"
        md_lines.append(line)

    md_content = "\n".join(md_lines)
    master_md_file = os.path.join(results_dir, "MASTER_EXPERIMENT_REPORT.md")
    with open(master_md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"[✓] Master Markdown raporu kaydedildi: {master_md_file}")
    print("\n" + "="*90)
    print("✅ TÜM SINIFLANDIRMA SUITE DENEYLERİ BAŞARIYLA TAMAMLANDI!")
    print("="*90)

if __name__ == "__main__":
    main()
