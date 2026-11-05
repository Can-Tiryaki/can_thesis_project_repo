import os
import json
import pandas as pd
import numpy as np

# Modülleri içe aktar
from prepare_data import get_data
from kfold_tuning_evaluator import evaluate_model_nested_kfold

# Model Arama Izgaralarını tanımlayalım
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():
    # Çalışma klasörlerini hazırla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    params_dir = os.path.join(base_dir, "model_parameters")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("[*] Özellikleri zenginleştirilmiş veri seti yükleniyor (Ağırlıklı K-Fold)...")
    X, y = get_data()
    print(f"-> Girdi Boyutu: {X.shape}, Hedef Boyutu: {y.shape}")

    # Test edilecek modellerin listesi ve arama ızgaraları
    models_config = [
        {
            'name': 'Random_Forest',
            'model': RandomForestRegressor(random_state=42),
            'grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [5, 10, 15],
                'max_features': ['sqrt', 0.6]
            }
        },
        {
            'name': 'Gradient_Boosting',
            'model': GradientBoostingRegressor(random_state=42),
            'grid': {
                'n_estimators': [100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [2, 3, 4],
                'min_samples_leaf': [5, 10, 15],
                'subsample': [0.7, 0.8, 0.9]
            }
        },
        {
            'name': 'XGBoost',
            'model': XGBRegressor(random_state=42),
            'grid': {
                'n_estimators': [100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [2, 3, 4],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0.1, 1.0, 5.0],
                'reg_lambda': [1.0, 5.0, 10.0]
            }
        },
        {
            'name': 'Artificial_Neural_Network',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(random_state=42, max_iter=500, solver='adam', activation='relu'))
            ]),
            'grid': {
                'mlp__hidden_layer_sizes': [(32,), (64, 32), (32, 16)],
                'mlp__alpha': [0.01, 0.1, 1.0, 5.0],
                'mlp__learning_rate_init': [0.001, 0.005]
            }
        },
        {
            'name': 'AdaBoost',
            'model': AdaBoostRegressor(random_state=42),
            'grid': {
                'estimator': [
                    DecisionTreeRegressor(max_depth=2, random_state=42),
                    DecisionTreeRegressor(max_depth=3, random_state=42),
                    DecisionTreeRegressor(max_depth=4, random_state=42)
                ],
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1, 0.5]
            }
        },
        {
            'name': 'Decision_Tree',
            'model': DecisionTreeRegressor(random_state=42),
            'grid': {
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [10, 20, 30],
                'min_samples_leaf': [5, 10, 15, 20]
            }
        }
    ]

    all_results = []

    print("\n[*] İç İçe (Nested) Gelişmiş Ağırlıklı K-Fold Çapraz Doğrulama Başlatılıyor...")
    
    for config in models_config:
        name = config['name']
        print(f"\n--> Model Optimize Ediliyor: {name}...")
        
        try:
            metrics = evaluate_model_nested_kfold(config['model'], config['grid'], X, y, n_splits=5)
            
            # Seçilen en iyi parametrelerin listesini kaydet
            param_file = os.path.join(params_dir, f"{name}_best_tuned_params.json")
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
                'Diyabetik_MAE_Val': metrics['mean_diabetic_mae_val'],
                'Durum': status
            }
            all_results.append(results_dict)
            print(f"   Optimizasyon Tamamlandı. Train R²: {metrics['mean_train_r2']:.4f} | Val R²: {metrics['mean_val_r2']:.4f} | Diyabetik Val MAE: {metrics['mean_diabetic_mae_val']:.4f}")
            
        except Exception as e:
            print(f"   Hata oluştu ({name}): {e}")
            import traceback
            traceback.print_exc()

    # Sonuçları DataFrame'e Dönüştür
    results_df = pd.DataFrame(all_results)
    
    # CSV Olarak Kaydet
    csv_file = os.path.join(results_dir, "tuned_comparison_results.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"\n[✓] Tüm sayısal tuning sonuçları CSV olarak kaydedildi: {csv_file}")

    # En İyi Modeli Bul
    best_idx = results_df['Val_R2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    print(f"\n🏆 En Başarılı Ağırlıklı Model: {best_model_name} (Ortalama Val R²: {results_df.loc[best_idx, 'Val_R2']:.4f})")

    # Markdown Tablosu Oluştur
    markdown_lines = []
    markdown_lines.append("# 📊 Hiperparametre Ayarlı ve Ağırlıklı Model Sonuçları (Gelişmiş Veri)")
    markdown_lines.append("Bu tablo, zenginleştirilmiş gelişmiş veri seti üzerinde **diyabetik hastalara 5 kat, prediyabetiklere 2 kat ceza ağırlığı uygulanarak** Nested 5-Fold Cross Validation ile elde edilen sonuçları göstermektedir.\n")
    markdown_lines.append("| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Ortalama MAE | Diyabetik MAE | Durum Analizi |")
    markdown_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for idx, row in results_df.iterrows():
        is_best = (idx == best_idx)
        
        def fmt(val):
            txt = f"{val:.4f}"
            return f"**{txt}**" if is_best else txt

        model_name_fmt = f"**{row['Model']}**" if is_best else row['Model']
        status_fmt = f"**{row['Durum']}**" if is_best else row['Durum']
        
        line = f"| {model_name_fmt} | {fmt(row['Train_R2'])} | {fmt(row['Val_R2'])} | {fmt(row['Overfitting_Gap_R2'])} | {fmt(row['Train_MSE'])} | {fmt(row['Val_MSE'])} | {fmt(row['Val_MAE'])} | {fmt(row['Diyabetik_MAE_Val'])} | {status_fmt} |"
        markdown_lines.append(line)

    markdown_content = "\n".join(markdown_lines)
    
    # Markdown Raporu Kaydet
    md_file = os.path.join(results_dir, "tuned_comparison_results.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"[✓] Markdown raporu kaydedildi: {md_file}")

    print("\n" + "="*95)
    print("📋 K-FOLD AĞIRLIKLI MODEL KARŞILAŞTIRMA TABLOSU (GELİŞMİŞ VERİ)")
    print("="*95)
    print(markdown_content)
    print("="*95)

if __name__ == "__main__":
    main()
