import os
import json
import pandas as pd
import numpy as np

# Modülleri içe aktar
from prepare_data import get_data
from kfold_evaluator import evaluate_model_kfold

# Model dosyalarını içe aktar
import model_rf
import model_gb
import model_xgb
import model_ann
import model_adaboost
import model_dt

def main():
    # Çalışma klasörlerini hazırla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    params_dir = os.path.join(base_dir, "model_parameters")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("[*] Özellikleri zenginleştirilmiş veri seti (engineered) yükleniyor...")
    X, y = get_data()
    print(f"-> Girdi Boyutu: {X.shape}, Hedef Boyutu: {y.shape}")

    # Test edilecek modellerin listesi
    model_modules = [
        model_rf,
        model_gb,
        model_xgb,
        model_ann,
        model_adaboost,
        model_dt
    ]

    all_results = []

    print("\n[*] 5-Fold Çapraz Doğrulama Deneyleri Başlatılıyor...")
    
    for module in model_modules:
        model, name, params = module.get_model()
        print(f"\n--> Model Eğitiliyor: {name}...")
        
        # 1. Hiperparametreleri JSON olarak kaydet
        param_file = os.path.join(params_dir, f"{name}_params.json")
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4, ensure_ascii=False)
        print(f"   Hiperparametreler kaydedildi: {param_file}")
        
        # 2. K-Fold Çapraz Doğrulama Çalıştır
        try:
            metrics = evaluate_model_kfold(model, X, y, n_splits=5)
            
            # Sonuçları listeye ekle
            results_dict = {
                'Model': name,
                'Mean_MSE': metrics['mean_mse'],
                'Std_MSE': metrics['std_mse'],
                'Mean_MAE': metrics['mean_mae'],
                'Std_MAE': metrics['std_mae'],
                'Mean_RMSE': metrics['mean_rmse'],
                'Std_RMSE': metrics['std_rmse'],
                'Mean_R2': metrics['mean_r2'],
                'Std_R2': metrics['std_r2']
            }
            all_results.append(results_dict)
            print(f"   Eğitim Tamamlandı. Ortalama R² Skoru: {metrics['mean_r2']:.4f}")
            
        except Exception as e:
            print(f"   Hata oluştu ({name}): {e}")

    # 3. Sonuçları DataFrame'e Dönüştür
    results_df = pd.DataFrame(all_results)
    
    # CSV Olarak Kaydet
    csv_file = os.path.join(results_dir, "comparison_results.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"\n[✓] Tüm sayısal sonuçlar CSV olarak kaydedildi: {csv_file}")

    # 4. En İyi Modeli Bul (Maksimum R² Alan Modeli En İyi Kabul Ediyoruz)
    best_idx = results_df['Mean_R2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    print(f"\n🏆 En Başarılı Model (Gelişmiş): {best_model_name} (Ortalama R²: {results_df.loc[best_idx, 'Mean_R2']:.4f})")

    # 5. Markdown Tablosu Oluştur ve En İyi Modeli Koyu Yap
    markdown_lines = []
    markdown_lines.append("# 📊 K-Fold Çapraz Doğrulama Model Karşılaştırma Sonuçları (Gelişmiş Veri)")
    markdown_lines.append("Bu tablo, özellik mühendisliği yapılmış zenginleştirilmiş veri seti (`mergeAllData_engineered.csv`) üzerinde 5-Fold Çapraz Doğrulama sonuçlarını karşılaştırmaktadır.\n")
    markdown_lines.append("| Model | Ortalama R² | R² Std | Ortalama MSE | MSE Std | Ortalama MAE | MAE Std | Ortalama RMSE | RMSE Std |")
    markdown_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for idx, row in results_df.iterrows():
        is_best = (idx == best_idx)
        
        # En başarılı modelin değerlerini kalın yapmak için wrapper
        def fmt(val, is_float=True):
            if is_float:
                txt = f"{val:.4f}"
            else:
                txt = str(val)
            return f"**{txt}**" if is_best else txt

        model_name_fmt = f"**{row['Model']}**" if is_best else row['Model']
        
        line = f"| {model_name_fmt} | {fmt(row['Mean_R2'])} | {fmt(row['Std_R2'])} | {fmt(row['Mean_MSE'])} | {fmt(row['Std_MSE'])} | {fmt(row['Mean_MAE'])} | {fmt(row['Std_MAE'])} | {fmt(row['Mean_RMSE'])} | {fmt(row['Std_RMSE'])} |"
        markdown_lines.append(line)

    markdown_content = "\n".join(markdown_lines)
    
    # Markdown Raporu Kaydet
    md_file = os.path.join(results_dir, "comparison_results.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"[✓] Markdown raporu kaydedildi: {md_file}")

    # 6. Sonuçları Konsola Bas
    print("\n" + "="*80)
    print("📋 K-FOLD MODEL KARŞILAŞTIRMA TABLOSU (GELİŞMİŞ VERİ)")
    print("="*80)
    print(markdown_content)
    print("="*80)

if __name__ == "__main__":
    main()
