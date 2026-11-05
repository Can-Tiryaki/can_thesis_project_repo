import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Model tanımları (Varsayılan Hiperparametreler)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Veriyi yükleme fonksiyonu
from prepare_data import get_data

def main():
    # Çalışma klasörlerini hazırla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(base_dir, "saved_models")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("[*] 100% Temiz normal veri seti yükleniyor (Varsayılan Parametreler)...")
    X, y = get_data()
    print(f"-> Girdi Boyutu: {X.shape}, Hedef Boyutu: {y.shape}")

    # 1. Tekil Tabakalı Bölme (Stratified Train-Test Split)
    # HbA1c kategorilerini geçici olarak oluşturup tabakalı bölüyoruz (%80 Train, %20 Test)
    y_binned = np.where(y < 5.7, 0, np.where(y < 6.5, 1, 2))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y_binned))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"-> Bölme tamamlandı. Train boyutu: {X_train.shape[0]}, Test boyutu: {X_test.shape[0]}")

    # 2. Varsayılan modelleri tanımlayalım
    models = {
        'Random_Forest': RandomForestRegressor(random_state=42),
        'Gradient_Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'Artificial_Neural_Network': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(random_state=42, max_iter=500))
        ]),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'Decision_Tree': DecisionTreeRegressor(random_state=42)
    }

    all_results = []
    trained_models = {}

    print("\n[*] Varsayılan Parametreli Modeller Eğitiliyor...")

    for name, model in models.items():
        print(f"--> Model Eğitiliyor: {name}...")
        
        # Modeli eğit
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Modeli joblib ile kaydet
        model_file = os.path.join(models_dir, f"{name}_default_model.pkl")
        joblib.dump(model, model_file)
        
        # Metrikleri hesapla
        # Eğitim Kümesi
        train_preds = model.predict(X_train)
        train_r2 = r2_score(y_train, train_preds)
        train_mse = mean_squared_error(y_train, train_preds)
        
        # Test Kümesi
        test_preds = model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(test_mse)
        
        gap = train_r2 - test_r2
        
        if gap > 0.25:
            status = "⚠️ Yüksek Overfitting"
        elif gap > 0.12:
            status = "🟡 Hafif Overfitting"
        elif gap < 0.02 and test_r2 < 0.20:
            status = "🔴 Underfitting"
        else:
            status = "🟢 Dengeli / Stabil Model"
            
        res = {
            'Model': name,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Overfitting_Gap_R2': gap,
            'Train_MSE': train_mse,
            'Test_MSE': test_mse,
            'Test_MAE': test_mae,
            'Test_RMSE': test_rmse,
            'Durum': status
        }
        all_results.append(res)
        print(f"   Eğitim R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Fark (Gap): {gap:.4f}")

    # DataFrame oluştur
    df_results = pd.DataFrame(all_results)
    
    # Sayısal sonuçları kaydet
    csv_path = os.path.join(results_dir, "stratified_comparison_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[✓] Sayısal sonuçlar CSV olarak kaydedildi: {csv_path}")

    # En başarılı test modelini bul
    best_idx = df_results['Test_R2'].idxmax()
    best_model_name = df_results.loc[best_idx, 'Model']
    best_fitted_model = trained_models[best_model_name]
    print(f"\n🏆 En Başarılı Model: {best_model_name} (Test R²: {df_results.loc[best_idx, 'Test_R2']:.4f})")

    # 4. Yan Yana Karşılaştırma Bar Grafiği Çiz (Train vs Test R²)
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(df_results))
    width = 0.35
    
    plt.bar(x_indices - width/2, df_results['Train_R2'], width, label='Eğitim (Train) R²', color='#3498db', edgecolor='black')
    plt.bar(x_indices + width/2, df_results['Test_R2'], width, label='Test (Test) R²', color='#e74c3c', edgecolor='black')
    
    plt.xlabel('Modeller', fontsize=12, fontweight='bold')
    plt.ylabel('R² Skoru', fontsize=12, fontweight='bold')
    plt.title('Tabakalı Bölme Sonrası Modellerin Train vs Test R² Skorları (Normal Veri - Varsayılan)', fontsize=14, fontweight='bold')
    plt.xticks(x_indices, df_results['Model'], rotation=15, fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    for idx, row in df_results.iterrows():
        plt.text(idx - width/2, row['Train_R2'] + 0.01, f"{row['Train_R2']:.2f}", ha='center', va='bottom', fontsize=9)
        plt.text(idx + width/2, row['Test_R2'] + 0.01, f"{row['Test_R2']:.2f}", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    chart_path = os.path.join(results_dir, "train_test_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[✓] Karşılaştırma bar grafiği kaydedildi: {chart_path}")

    # 5. En Başarılı Model İçin Saçılım Grafiği (Predicted vs Actual) Çiz
    best_test_preds = best_fitted_model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_test_preds, alpha=0.7, color='#2ecc71', edgecolor='black', label='Hastalar')
    lims = [min(min(y_test), min(best_test_preds)), max(max(y_test), max(best_test_preds))]
    plt.plot(lims, lims, color='red', linestyle='--', linewidth=2, label='Mükemmel Tahmin Referansı')
    
    plt.xlabel('Gerçek HbA1c (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Tahmin Edilen HbA1c (%)', fontsize=12, fontweight='bold')
    plt.title(f'En Başarılı Model ({best_model_name}) Tahmin Saçılım Grafiği (Varsayılan)\nTest R²: {df_results.loc[best_idx, "Test_R2"]:.4f}', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    predictions_path = os.path.join(results_dir, "best_model_predictions.png")
    plt.savefig(predictions_path, dpi=150)
    plt.close()
    print(f"[✓] Tahmin saçılım grafiği kaydedildi: {predictions_path}")

    # 6. En Başarılı Model İçin Özellik Önem Sırası (Feature Importance) Grafiği Çiz (Ağaç tabanlıysa)
    if hasattr(best_fitted_model, 'feature_importances_') or (hasattr(best_fitted_model, 'steps') and hasattr(best_fitted_model.steps[-1][1], 'feature_importances_')):
        if hasattr(best_fitted_model, 'feature_importances_'):
            importances = best_fitted_model.feature_importances_
        else:
            importances = best_fitted_model.steps[-1][1].feature_importances_
            
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis')
        plt.xlabel('Özellik Skoru (Gini/Gain Importance)', fontsize=12, fontweight='bold')
        plt.ylabel('Girdi Değişkenleri (Features)', fontsize=12, fontweight='bold')
        plt.title(f'En Başarılı Model ({best_model_name}) İçin Özellik Önem Sıralaması (Varsayılan)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        importance_path = os.path.join(results_dir, "best_model_feature_importance.png")
        plt.savefig(importance_path, dpi=150)
        plt.close()
        print(f"[✓] Özellik önem sırası grafiği kaydedildi: {importance_path}")

    # 7. Markdown Tablosu Oluştur ve En İyi Modeli Koyu Yap
    markdown_lines = []
    markdown_lines.append("# 📊 Tekil Tabakalı Bölme Model Sonuçları (Normal Veri - Varsayılan)")
    markdown_lines.append("Bu tablo, temizlenmiş normal veri setinin (%80 Eğitim, %20 Test) olarak tabakalı şekilde tekil olarak bölünmesi ve modellerin **varsayılan (default)** parametrelerle eğitilmesiyle elde edilen test seti başarılarını göstermektedir.\n")
    markdown_lines.append("| Model | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Eğitim MSE | Test MSE | Test MAE | Test RMSE | Durum Analizi |")
    markdown_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for idx, row in df_results.iterrows():
        is_best = (idx == best_idx)
        
        def fmt(val):
            txt = f"{val:.4f}"
            return f"**{txt}**" if is_best else txt

        model_name_fmt = f"**{row['Model']}**" if is_best else row['Model']
        status_fmt = f"**{row['Durum']}**" if is_best else row['Durum']
        
        line = f"| {model_name_fmt} | {fmt(row['Train_R2'])} | {fmt(row['Test_R2'])} | {fmt(row['Overfitting_Gap_R2'])} | {fmt(row['Train_MSE'])} | {fmt(row['Test_MSE'])} | {fmt(row['Test_MAE'])} | {fmt(row['Test_RMSE'])} | {status_fmt} |"
        markdown_lines.append(line)

    markdown_content = "\n".join(markdown_lines)
    
    # Markdown Raporu Kaydet
    md_file = os.path.join(results_dir, "stratified_comparison_results.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"[✓] Markdown raporu kaydedildi: {md_file}")

    # Sonuçları Konsola Bas
    print("\n" + "="*95)
    print("📋 TABAKALI BÖLME MODEL KARŞILAŞTIRMA TABLOSU (NORMAL VERİ - VARSAYILAN)")
    print("="*95)
    print(markdown_content)
    print("="*95)

if __name__ == "__main__":
    main()
