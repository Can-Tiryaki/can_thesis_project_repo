import os
import json
import pandas as pd
import numpy as np

# Modülleri içe aktar
from prepare_data import get_data
from kfold_tuning_evaluator import evaluate_model_nested_kfold

# Model Grid dosyalarını içe aktar
import grid_rf
import grid_gb
import grid_xgb
import grid_ann
import grid_adaboost
import grid_dt

def main():
    # Çalışma klasörlerini hazırla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    params_dir = os.path.join(base_dir, "model_parameters")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("[*] Özellikleri zenginleştirilmiş veri seti yükleniyor (Tuning Aşaması)...")
    X, y = get_data()
    print(f"-> Girdi Boyutu: {X.shape}, Hedef Boyutu: {y.shape}")

    # Test edilecek modellerin listesi
    grid_modules = [
        grid_rf,
        grid_gb,
        grid_xgb,
        grid_ann,
        grid_adaboost,
        grid_dt
    ]

    all_results = []

    print("\n[*] İç İçe (Nested) K-Fold ile Hiperparametre Arama & Değerlendirme Başlatılıyor (Gelişmiş Veri)...")
    
    for module in grid_modules:
        model, param_grid, name = module.get_grid()
        print(f"\n--> Model Optimize Ediliyor ve Değerlendiriliyor: {name}...")
        
        # Nested K-Fold ile eğitimi ve aramayı koştur
        try:
            metrics = evaluate_model_nested_kfold(model, param_grid, X, y, n_splits=5)
            
            # Seçilen en iyi parametrelerin listesini kaydet
            param_file = os.path.join(params_dir, f"{name}_best_tuned_params.json")
            # JSON formatında kaydederken nested/custom estimator'ları string'e çevirmek için custom serializer kullanıyoruz
            def custom_serializer(obj):
                if hasattr(obj, '__class__'):
                    return obj.__class__.__name__
                return str(obj)
                
            with open(param_file, "w", encoding="utf-8") as f:
                json.dump(metrics['best_params_list'], f, indent=4, ensure_ascii=False, default=custom_serializer)
            print(f"   En iyi parametreler fold bazında kaydedildi: {param_file}")
            
            gap = metrics['mean_train_r2'] - metrics['mean_val_r2']
            
            # Durum tespiti
            if gap > 0.25:
                status = "⚠️ Yüksek Overfitting"
            elif gap > 0.12:
                status = "🟡 Hafif Overfitting"
            elif gap < 0.02 and metrics['mean_val_r2'] < 0.20:
                status = "🔴 Underfitting"
            else:
                status = "🟢 Dengeli / Stabil Model"
            
            # Sonuçları listeye ekle
            results_dict = {
                'Model': name,
                'Train_R2': metrics['mean_train_r2'],
                'Val_R2': metrics['mean_val_r2'],
                'Overfitting_Gap_R2': gap,
                'Train_MSE': metrics['mean_train_mse'],
                'Val_MSE': metrics['mean_val_mse'],
                'Val_MAE': metrics['mean_val_mae'],
                'Durum': status
            }
            all_results.append(results_dict)
            print(f"   Optimizasyon Tamamlandı. Train R²: {metrics['mean_train_r2']:.4f} | Val R²: {metrics['mean_val_r2']:.4f} | Fark: {gap:.4f}")
            
        except Exception as e:
            print(f"   Hata oluştu ({name}): {e}")
            import traceback
            traceback.print_exc()

    # 3. Sonuçları DataFrame'e Dönüştür
    results_df = pd.DataFrame(all_results)
    
    # CSV Olarak Kaydet
    csv_file = os.path.join(results_dir, "tuned_comparison_results.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"\n[✓] Tüm sayısal tuning sonuçları CSV olarak kaydedildi: {csv_file}")

    # 4. En İyi Modeli Bul (Maksimum Validation R² Alan Modeli En İyi Kabul Ediyoruz)
    best_idx = results_df['Val_R2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    print(f"\n🏆 En Başarılı Tuned Model (Gelişmiş): {best_model_name} (Ortalama Val R²: {results_df.loc[best_idx, 'Val_R2']:.4f})")

    # 5. Markdown Tablosu Oluştur ve En İyi Modeli Koyu Yap
    markdown_lines = []
    markdown_lines.append("# 📊 Hiperparametre Ayarlı (Tuned) Model Karşılaştırma Sonuçları (Gelişmiş Veri)")
    markdown_lines.append("Bu tablo, özellik mühendisliği yapılmış zenginleştirilmiş veri seti (`mergeAllData_engineered.csv`) üzerinde **GridSearchCV + Nested 5-Fold Cross Validation** kullanılarak optimize edilen modellerin sonuçlarını göstermektedir.\n")
    markdown_lines.append("| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Ortalama MAE | Durum Analizi |")
    markdown_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for idx, row in results_df.iterrows():
        is_best = (idx == best_idx)
        
        def fmt(val):
            txt = f"{val:.4f}"
            return f"**{txt}**" if is_best else txt

        model_name_fmt = f"**{row['Model']}**" if is_best else row['Model']
        status_fmt = f"**{row['Durum']}**" if is_best else row['Durum']
        
        line = f"| {model_name_fmt} | {fmt(row['Train_R2'])} | {fmt(row['Val_R2'])} | {fmt(row['Overfitting_Gap_R2'])} | {fmt(row['Train_MSE'])} | {fmt(row['Val_MSE'])} | {fmt(row['Val_MAE'])} | {status_fmt} |"
        markdown_lines.append(line)

    markdown_content = "\n".join(markdown_lines)
    
    # Markdown Raporu Kaydet
    md_file = os.path.join(results_dir, "tuned_comparison_results.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"[✓] Markdown raporu kaydedildi: {md_file}")

    # 6. Sonuçları Konsola Bas
    print("\n" + "="*95)
    print("📋 K-FOLD TUNED MODEL KARŞILAŞTIRMA TABLOSU (GELİŞMİŞ VERİ)")
    print("="*95)
    print(markdown_content)
    print("="*95)

if __name__ == "__main__":
    main()
