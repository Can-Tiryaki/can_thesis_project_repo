import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")

def main():
    base_dir = PROJECT_ROOT
    
    csv_normal_kfold = os.path.join(base_dir, "models/k_fold_weighted/normal/results/tuned_comparison_results.csv")
    csv_eng_kfold = os.path.join(base_dir, "models/k_fold_weighted/feature_engineered/results/tuned_comparison_results.csv")
    
    paths = [csv_normal_kfold, csv_eng_kfold]
    for p in paths:
        if not os.path.exists(p):
            print(f"Hata: Dosya bulunamadı: {p}")
            return
            
    df_nk = pd.read_csv(csv_normal_kfold)
    df_ek = pd.read_csv(csv_eng_kfold)
    
    rows = []
    
    for _, r in df_nk.iterrows():
        rows.append({
            'Model': r['Model'],
            'Senaryo': '5. Temel Veri (Weighted - K-Fold)',
            'Train_R2': r['Train_R2'],
            'Val_R2': r['Val_R2'],
            'Gap_R2': r['Overfitting_Gap_R2'],
            'Val_MSE': r['Val_MSE'],
            'Diyabetik_MAE_Val': r['Diyabetik_MAE_Val']
        })
        
    for _, r in df_ek.iterrows():
        rows.append({
            'Model': r['Model'],
            'Senaryo': '6. Gelişmiş Veri (Weighted - K-Fold)',
            'Train_R2': r['Train_R2'],
            'Val_R2': r['Val_R2'],
            'Gap_R2': r['Overfitting_Gap_R2'],
            'Val_MSE': r['Val_MSE'],
            'Diyabetik_MAE_Val': r['Diyabetik_MAE_Val']
        })
        
    master_df = pd.DataFrame(rows)
    
    master_csv = os.path.join(base_dir, "analysis/kfold_weighted_master_comparison.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"[✓] K-Fold Ağırlıklı Master karşılaştırma tablosu kaydedildi: {master_csv}")
    
    best_indices = []
    for s in master_df['Senaryo'].unique():
        sub_df = master_df[master_df['Senaryo'] == s]
        best_indices.append(sub_df['Val_R2'].idxmax())
        
    md_lines = []
    md_lines.append("# 📊 K-Fold Çapraz Doğrulama Ağırlıklı Bölme Master Karşılaştırma Sonuçları")
    md_lines.append("Bu tablo; Nested 5-Fold Cross Validation bölünmesi sırasında diyabetik hastalara 5 kat, prediyabetiklere 2 kat ceza ağırlığı verildiğinde elde edilen ortalama train/validation sonuçlarını karşılaştırmaktadır.\n")
    md_lines.append("| Model | Deney Senaryosu | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Doğrulama MSE | Diyabetik MAE | Durum Analizi |")
    md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    for idx, row in master_df.iterrows():
        is_best_in_scenario = (idx in best_indices)
        gap = row['Gap_R2']
        
        if gap > 0.25:
            status = "⚠️ Yüksek Overfitting"
        elif gap > 0.12:
            status = "🟡 Hafif Overfitting"
        elif gap < 0.02 and row['Val_R2'] < 0.20:
            status = "🔴 Underfitting"
        else:
            status = "🟢 Dengeli / Stabil Model"
            
        def fmt(val):
            txt = f"{val:.4f}"
            return f"**{txt}**" if is_best_in_scenario else txt
            
        model_fmt = f"**{row['Model']}**" if is_best_in_scenario else row['Model']
        scenario_fmt = f"**{row['Senaryo']}**" if is_best_in_scenario else row['Senaryo']
        status_fmt = f"**{status}**" if is_best_in_scenario else status
        
        line = f"| {model_fmt} | {scenario_fmt} | {fmt(row['Train_R2'])} | {fmt(row['Val_R2'])} | {fmt(gap)} | {fmt(row['Val_MSE'])} | {fmt(row['Diyabetik_MAE_Val'])} | {status_fmt} |"
        md_lines.append(line)
        
    md_content = "\n".join(md_lines)
    
    md_path = os.path.join(base_dir, "analysis/kfold_weighted_master_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[✓] K-Fold Ağırlıklı Master Markdown raporu kaydedildi: {md_path}")
    
    print("\n" + "="*110)
    print("📋 TÜM K-FOLD AĞIRLIKLI DENEYLERİN MASTER KARŞILAŞTIRMA TABLOSU")
    print("="*110)
    print(md_content)
    print("="*110)

if __name__ == "__main__":
    main()
