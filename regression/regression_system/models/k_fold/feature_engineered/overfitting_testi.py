import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score

# Modülleri yükle
from prepare_data import get_data
import model_rf
import model_gb
import model_xgb
import model_ann
import model_adaboost
import model_dt

def run_overfitting_analysis():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("[*] Veri seti (gelişmiş) yükleniyor...")
    X, y = get_data()
    
    # 5-Fold Stratified Split için HbA1c kategorilerini hazırla
    y_binned = np.where(y < 5.7, 0, np.where(y < 6.5, 1, 2))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model_modules = [
        model_rf,
        model_gb,
        model_xgb,
        model_ann,
        model_adaboost,
        model_dt
    ]
    
    overfitting_results = []
    
    print("\n[*] Aşırı Öğrenme (Overfitting) Testi Başlatılıyor (Gelişmiş Veri)...")
    
    for module in model_modules:
        model, name, _ = module.get_model()
        print(f"--> {name} test ediliyor...")
        
        train_r2s, val_r2s = [], []
        train_mses, val_mses = [], []
        
        for train_idx, val_idx in skf.split(X, y_binned):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Modeli eğit
            model.fit(X_train, y_train)
            
            # Eğitim kümesi tahminleri ve değerlendirme
            train_preds = model.predict(X_train)
            train_r2s.append(r2_score(y_train, train_preds))
            train_mses.append(mean_squared_error(y_train, train_preds))
            
            # Doğrulama (validation) kümesi tahminleri ve değerlendirme
            val_preds = model.predict(X_val)
            val_r2s.append(r2_score(y_val, val_preds))
            val_mses.append(mean_squared_error(y_val, val_preds))
            
        mean_train_r2 = np.mean(train_r2s)
        mean_val_r2 = np.mean(val_r2s)
        gap = mean_train_r2 - mean_val_r2
        
        res = {
            'Model': name,
            'Train_R2': mean_train_r2,
            'Train_R2_Std': np.std(train_r2s),
            'Val_R2': mean_val_r2,
            'Val_R2_Std': np.std(val_r2s),
            'Train_MSE': np.mean(train_mses),
            'Val_MSE': np.mean(val_mses),
            'Overfitting_Gap_R2': gap
        }
        overfitting_results.append(res)
        print(f"   Eğitim R²: {mean_train_r2:.4f} | Doğrulama R²: {mean_val_r2:.4f} | Fark: {gap:.4f}")
        
    # DataFrame oluştur
    df_results = pd.DataFrame(overfitting_results)
    
    # Sonuçları CSV olarak kaydet
    csv_path = os.path.join(results_dir, "overfitting_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[✓] Sonuçlar CSV olarak kaydedildi: {csv_path}")
    
    # En başarılı test skoruna sahip modeli koyu yapalım
    best_idx = df_results['Val_R2'].idxmax()
    
    # 1. Yan Yana Karşılaştırma Bar Grafiği Oluştur
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(df_results))
    width = 0.35
    
    plt.bar(x_indices - width/2, df_results['Train_R2'], width, label='Eğitim (Train) R²', color='#3498db', edgecolor='black')
    plt.bar(x_indices + width/2, df_results['Val_R2'], width, label='Doğrulama (Val) R²', color='#e74c3c', edgecolor='black')
    
    plt.xlabel('Modeller', fontsize=12, fontweight='bold')
    plt.ylabel('R² Skoru', fontsize=12, fontweight='bold')
    plt.title('Modellerin Eğitim vs Doğrulama Başarımı ve Overfitting Analizi (Gelişmiş Veri)', fontsize=14, fontweight='bold')
    plt.xticks(x_indices, df_results['Model'], rotation=15, fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Barların üstüne değer yazdıralım
    for idx, row in df_results.iterrows():
        plt.text(idx - width/2, row['Train_R2'] + 0.01, f"{row['Train_R2']:.2f}", ha='center', va='bottom', fontsize=9)
        plt.text(idx + width/2, row['Val_R2'] + 0.01, f"{row['Val_R2']:.2f}", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    img_path = os.path.join(results_dir, "overfitting_comparison.png")
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"[✓] Overfitting karşılaştırma grafiği kaydedildi: {img_path}")
    
    # 2. Markdown Tablosu Hazırla
    md_lines = []
    md_lines.append("# 🔍 Aşırı Öğrenme (Overfitting) Analiz Raporu (Gelişmiş Veri)")
    md_lines.append("Bu rapor, özellik mühendisliği yapılmış zenginleştirilmiş veri seti (`mergeAllData_engineered.csv`) üzerinde modellerin aşırı öğrenme (overfitting) durumunu, **Eğitim Seti (Train)** ve **Doğrulama Seti (Validation)** sonuçlarını karşılaştırarak analiz eder.\n")
    md_lines.append("### Temel Analiz Bulguları:")
    md_lines.append("*   **Overfitting Gap:** Eğitim R² ile Doğrulama R² arasındaki farktır. Bu farkın düşük olması modelin daha genelleyici ve kararlı olduğunu gösterir.")
    md_lines.append("*   En iyi performansı gösteren model **koyu** harflerle işaretlenmiştir.\n")
    
    md_lines.append("| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Durum Analizi |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    for idx, row in df_results.iterrows():
        is_best = (idx == best_idx)
        gap = row['Overfitting_Gap_R2']
        
        # Durum tespiti
        if gap > 0.25:
            status = "⚠️ Yüksek Overfitting (Ezberleme)"
        elif gap > 0.12:
            status = "🟡 Hafif Overfitting"
        elif gap < 0.02 and row['Val_R2'] < 0.20:
            status = "🔴 Underfitting (Yetersiz Öğrenme)"
        else:
            status = "🟢 Dengeli / Stabil Model"
            
        def fmt(val):
            txt = f"{val:.4f}"
            return f"**{txt}**" if is_best else txt
            
        name_fmt = f"**{row['Model']}**" if is_best else row['Model']
        status_fmt = f"**{status}**" if is_best else status
        
        line = f"| {name_fmt} | {fmt(row['Train_R2'])} | {fmt(row['Val_R2'])} | {fmt(gap)} | {fmt(row['Train_MSE'])} | {fmt(row['Val_MSE'])} | {status_fmt} |"
        md_lines.append(line)
        
    md_content = "\n".join(md_lines)
    
    md_path = os.path.join(results_dir, "overfitting_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[✓] Rapor kaydedildi: {md_path}")
    
    print("\n" + "="*90)
    print("📋 OVERFITTING KARŞILAŞTIRMA TABLOSU (GELİŞMİŞ VERİ)")
    print("="*90)
    print(md_content)
    print("="*90)

if __name__ == "__main__":
    run_overfitting_analysis()
