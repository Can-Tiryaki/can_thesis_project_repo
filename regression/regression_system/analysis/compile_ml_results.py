import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")

def main():
    # Dosya yolları
    base_dir = PROJECT_ROOT
    
    csv_normal_base = os.path.join(base_dir, "models/k_fold/normal/results/overfitting_results.csv")
    csv_eng_base = os.path.join(base_dir, "models/k_fold/feature_engineered/results/overfitting_results.csv")
    csv_normal_tuned = os.path.join(base_dir, "models/hyperparameter_tuned/normal/results/tuned_comparison_results.csv")
    csv_eng_tuned = os.path.join(base_dir, "models/hyperparameter_tuned/feature_engineered/results/tuned_comparison_results.csv")
    
    # Dosyaların varlığını kontrol et
    paths = [csv_normal_base, csv_eng_base, csv_normal_tuned, csv_eng_tuned]
    for p in paths:
        if not os.path.exists(p):
            print(f"Hata: Dosya bulunamadı: {p}")
            return
            
    # Verileri yükle
    df_nb = pd.read_csv(csv_normal_base)
    df_eb = pd.read_csv(csv_eng_base)
    df_nt = pd.read_csv(csv_normal_tuned)
    df_et = pd.read_csv(csv_eng_tuned)
    
    # Karşılaştırma listesi hazırlayalım
    # Sütunlar: Model, Deney Senaryosu, Train R², Val R², Overfitting Gap, Train MSE, Val MSE
    rows = []
    
    # Senaryoları ekle
    # 1. Normal Base
    for _, r in df_nb.iterrows():
        rows.append({
            'Model': r['Model'],
            'Senaryo': '1. Temel Veri (Varsayılan)',
            'Train_R2': r['Train_R2'],
            'Val_R2': r['Val_R2'],
            'Gap_R2': r['Overfitting_Gap_R2'],
            'Val_MSE': r['Val_MSE']
        })
        
    # 2. Engineered Base
    for _, r in df_eb.iterrows():
        rows.append({
            'Model': r['Model'],
            'Senaryo': '2. Gelişmiş Veri (Varsayılan)',
            'Train_R2': r['Train_R2'],
            'Val_R2': r['Val_R2'],
            'Gap_R2': r['Overfitting_Gap_R2'],
            'Val_MSE': r['Val_MSE']
        })
        
    # 3. Normal Tuned
    for _, r in df_nt.iterrows():
        rows.append({
            'Model': r['Model'],
            'Senaryo': '3. Temel Veri (Hiperparametre Ayarlı)',
            'Train_R2': r['Train_R2'],
            'Val_R2': r['Val_R2'],
            'Gap_R2': r['Overfitting_Gap_R2'],
            'Val_MSE': r['Val_MSE']
        })
        
    # 4. Engineered Tuned
    for _, r in df_et.iterrows():
        rows.append({
            'Model': r['Model'],
            'Senaryo': '4. Gelişmiş Veri (Hiperparametre Ayarlı)',
            'Train_R2': r['Train_R2'],
            'Val_R2': r['Val_R2'],
            'Gap_R2': r['Overfitting_Gap_R2'],
            'Val_MSE': r['Val_MSE']
        })
        
    master_df = pd.DataFrame(rows)
    
    # Master tabloyu CSV olarak kaydet
    master_csv = os.path.join(base_dir, "analysis/ml_master_comparison.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"[✓] Master karşılaştırma tablosu kaydedildi: {master_csv}")
    
    # Her senaryo için en iyi Val R² değerini bulup kalın yapalım
    # Senaryo bazında en yüksek Val R² veren satırı belirleyelim
    best_indices = []
    for s in master_df['Senaryo'].unique():
        sub_df = master_df[master_df['Senaryo'] == s]
        best_indices.append(sub_df['Val_R2'].idxmax())
        
    # Markdown çıktısı oluştur
    md_lines = []
    md_lines.append("# 🏆 Tüm Deneylerin Master Karşılaştırma Sonuçları (ML Master Table)")
    md_lines.append("Bu tablo; ham veri, özellik mühendisliği yapılmış veri ve hiperparametre ayarlarının (tuning) model performansları ile overfitting durumları üzerindeki etkilerini bir arada göstermektedir.\n")
    md_lines.append("| Model | Deney Senaryosu | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Doğrulama MSE | Durum Analizi |")
    md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
    
    for idx, row in master_df.iterrows():
        is_best_in_scenario = (idx in best_indices)
        gap = row['Gap_R2']
        
        # Durum tespiti
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
        
        line = f"| {model_fmt} | {scenario_fmt} | {fmt(row['Train_R2'])} | {fmt(row['Val_R2'])} | {fmt(gap)} | {fmt(row['Val_MSE'])} | {status_fmt} |"
        md_lines.append(line)
        
    md_content = "\n".join(md_lines)
    
    md_path = os.path.join(base_dir, "analysis/ml_master_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[✓] Master Markdown raporu kaydedildi: {md_path}")
    
    print("\n" + "="*110)
    print("📋 TÜM DENEYLERİN MASTER KARŞILAŞTIRMA TABLOSU")
    print("="*110)
    print(md_content)
    print("="*110)

if __name__ == "__main__":
    main()
