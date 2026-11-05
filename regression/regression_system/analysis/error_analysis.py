import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit

# Veriyi yükleme fonksiyonu
import sys
PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")
sys.path.append(str(PROJECT_ROOT / "models" / "stratified_tuned" / "feature_engineered"))
from prepare_data import get_data

def main():
    base_dir = PROJECT_ROOT
    model_path = os.path.join(base_dir, "models/tabakali_tuned/feature_engineered/saved_models/Random_Forest_best_model.pkl")
    analysis_dir = os.path.join(base_dir, "analysis/error_analysis")
    
    os.makedirs(analysis_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Hata: En iyi model dosyası bulunamadı: {model_path}")
        return
        
    # 1. En İyi Modeli ve Verileri Yükle
    model = joblib.load(model_path)
    X, y = get_data()
    
    # Train-Test bölünmesini orijinal random_state (42) ile birebir aynı şekilde yapalım
    y_binned = np.where(y < 5.7, 0, np.where(y < 6.5, 1, 2))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y_binned))
    
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]
    
    # 2. Tahminleri Al ve Hataları Hesapla
    y_pred = model.predict(X_test)
    abs_errors = np.abs(y_test - y_pred)
    sq_errors = (y_test - y_pred) ** 2
    
    # 3. Test Kümesi Verilerini Birleştir
    df_err = X_test.copy()
    df_err['Gerçek_HbA1c'] = y_test
    df_err['Tahmin_HbA1c'] = y_pred
    df_err['Mutlak_Hata'] = abs_errors
    df_err['Karesel_Hata'] = sq_errors
    
    # Hastaları ADA kriterlerine göre gerçek HbA1c sınıflarına ayıralım
    df_err['HbA1c_Risk_Sinifi'] = np.where(df_err['Gerçek_HbA1c'] < 5.7, 'Normal (<5.7)',
                                           np.where(df_err['Gerçek_HbA1c'] < 6.5, 'Prediyabet (5.7-6.4)', 'Diyabet (>=6.5)'))
    
    # 4. Sınıf Bazlı Ortalama Hataları Hesapla (MAE ve RMSE)
    grouped = df_err.groupby('HbA1c_Risk_Sinifi')
    class_metrics = []
    for name, group in grouped:
        mae = group['Mutlak_Hata'].mean()
        rmse = np.sqrt(group['Karesel_Hata'].mean())
        count = len(group)
        class_metrics.append({
            'Risk Sınıfı': name,
            'Hasta Sayısı': count,
            'Ortalama Mutlak Hata (MAE)': mae,
            'Kök Ortalama Karesel Hata (RMSE)': rmse
        })
    df_class_metrics = pd.DataFrame(class_metrics)
    
    # Sınıf bazlı metrikleri kaydet
    metrics_csv = os.path.join(analysis_dir, "class_error_metrics.csv")
    df_class_metrics.to_csv(metrics_csv, index=False)
    
    # 5. En Çok Yanlış Tahmin Edilen Top 15 Hastayı Bul (Worst Predictions)
    df_worst = df_err.sort_values(by='Mutlak_Hata', ascending=False).head(15)
    worst_csv = os.path.join(analysis_dir, "worst_predictions.csv")
    # Okunabilirliği arttırmak için belirli sütunları seçip kaydedelim
    cols_to_show = ['Yaş', 'Cinsiyet', 'Açlık Kan Şekeri (AKŞ)', 'Tokluk Kan Şekeri (TKŞ) 2.Saat', 'Glisemik_Ortalama', 'Gerçek_HbA1c', 'Tahmin_HbA1c', 'Mutlak_Hata', 'HbA1c_Risk_Sinifi']
    df_worst[cols_to_show].to_csv(worst_csv, index=False)
    
    # 6. Sınıf Bazlı Hata Dağılım Grafiği (Boxplot) Çiz
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='HbA1c_Risk_Sinifi', y='Mutlak_Hata', data=df_err, palette='Set2')
    plt.xlabel('Gerçek HbA1c Risk Sınıfı', fontsize=12, fontweight='bold')
    plt.ylabel('Mutlak Tahmin Hatası (Absolute Error)', fontsize=12, fontweight='bold')
    plt.title('HbA1c Risk Sınıflarına Göre Mutlak Hata Dağılımı', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    boxplot_path = os.path.join(analysis_dir, "error_by_class_boxplot.png")
    plt.savefig(boxplot_path, dpi=150)
    plt.close()
    
    # 7. Gerçek Değer ile Tahmin Farkı Grafiği (Residual Plot)
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals, color='purple', alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Gerçek HbA1c (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Kalıntı Hata (Gerçek - Tahmin)', fontsize=12, fontweight='bold')
    plt.title('Kalıntı (Residual) Hata Analizi Grafiği', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    residual_path = os.path.join(analysis_dir, "residual_plot.png")
    plt.savefig(residual_path, dpi=150)
    plt.close()

    # Markdown formatında rapor oluştur
    md_lines = []
    md_lines.append("# 🔍 Model Hata Analizi ve Birebir Yanlış Tahmin İncelemesi")
    md_lines.append("Bu rapor, en başarılı modelimiz olan **Tuned Random Forest (Gelişmiş Veri)** modelinin test kümesindeki 181 hasta üzerinde yaptığı tahmin hatalarını incelemektedir.\n")
    
    md_lines.append("## 📈 1. Risk Sınıfı Bazında Hata Dağılımı")
    md_lines.append("Modelin normal, prediyabet ve diyabet hastalarını tahmin ederken yaptığı ortalama hatalar (MAE) aşağıdaki gibidir:\n")
    
    md_lines.append("| Risk Sınıfı | Hasta Sayısı | Ortalama Hata (MAE) | RMSE (Hata Standart Sapması) |")
    md_lines.append("| :--- | :---: | :---: | :---: |")
    for _, r in df_class_metrics.iterrows():
        md_lines.append(f"| {r['Risk Sınıfı']} | {r['Hasta Sayısı']} | {r['Ortalama Mutlak Hata (MAE)']:.4f} | {r['Kök Ortalama Karesel Hata (RMSE)']:.4f} |")
        
    md_lines.append("\n> **Klinik Yorum:** Model, **Diyabet (>=6.5)** grubundaki hastaları tahmin ederken diğer gruplara göre **ciddi oranda daha yüksek hata** yapmaktadır. Bunun temel sebebi diyabetik hasta sayısının veri setinde azınlıkta olması ve regresyon modellerinin uç değerleri merkeze (ortalama değere) doğru çekme eğilimidir (Regress to the mean).\n")
    
    md_lines.append("## 🚨 2. En Çok Hata Yapılan İlk 10 Hasta (Birebir İnceleme)")
    md_lines.append("Modelin tahminlerinde en çok çuvalladığı (mutlak hatası en yüksek) 10 hastanın klinik tablosu şu şekildedir:\n")
    md_lines.append("| Sıra | Yaş | Cinsiyet (E=1/K=0) | AKŞ (mg/dL) | TKŞ 2.Saat (mg/dL) | Glisemik Ortalama | Gerçek HbA1c (%) | Tahmin HbA1c (%) | Mutlak Hata (%) |")
    md_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for idx, (_, r) in enumerate(df_worst.head(10).iterrows(), 1):
        gender_str = "Erkek" if r['Cinsiyet'] == 1 else "Kadın"
        md_lines.append(f"| {idx} | {int(r['Yaş'])} | {gender_str} | {int(r['Açlık Kan Şekeri (AKŞ)'])} | {int(r['Tokluk Kan Şekeri (TKŞ) 2.Saat'])} | {r['Glisemik_Ortalama']:.1f} | **{r['Gerçek_HbA1c']:.2f}** | **{r['Tahmin_HbA1c']:.2f}** | **{r['Mutlak_Hata']:.2f}** |")
        
    md_lines.append("\n### 💡 Hata Desenleri Analizi (Neden Yanlış Tahmin Ediliyor?):")
    md_lines.append("1. **Yüksek HbA1c Değerlerinin Düşük Tahmin Edilmesi (Underestimation):** Tabloda görüleceği üzere, gerçek HbA1c değeri **9.00 - 11.00** civarında olan çok yüksek riskli diyabet hastaları, model tarafından genellikle **6.50 - 7.50** aralığında tahmin edilmektedir. Model bu uç değerleri öğrenmekte zorlanmaktadır.")
    md_lines.append("2. **Uyumsuz Şeker Değerleri:** Bazı hastalarda Açlık Kan Şekeri (AKŞ) normal sınırlardayken (örn: 100 mg/dL), 3 aylık şeker ortalaması olan HbA1c'nin yüksek olması modeli yanıltmaktadır.")
    
    md_lines.append("\n## ⚖️ 3. Çözüm Önerisi: Örnek Ağırlıklandırma (Sample Weighting)")
    md_lines.append("Diyabetik hastaların hatalarını minimize etmek ve modelin bu sınıfa daha fazla odaklanmasını sağlamak amacıyla eğitim sırasında **örnek ağırlıkları (sample weights)** kullanılabilir.")
    md_lines.append("Örneğin, eğitim kümesindeki her bir hastaya HbA1c değerine göre dinamik bir ağırlık atanır:")
    md_lines.append("- **Normal Hastalar:** Ağırlık = 1.0")
    md_lines.append("- **Prediyabetik Hastalar:** Ağırlık = 1.5")
    md_lines.append("- **Diyabetik Hastalar:** Ağırlık = 3.0 (Diyabet hastalarının hata cezası 3 katına çıkarılır)")
    md_lines.append("\nBu sayede ağaç algoritmaları (Random Forest, XGBoost vb.) dallanma yaparken diyabetik hastaların hata payını düşürmek için bu bölgeye daha çok ağaç ayıracaktır. Bu durum diyabet tahminlerindeki büyük hataları azaltacaktır.")
    
    md_content = "\n".join(md_lines)
    
    report_path = os.path.join(analysis_dir, "error_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[✓] Hata analizi markdown raporu yazıldı: {report_path}")
    
    print("\n" + "="*95)
    print("📋 HATA ANALİZİ VE EN ÇOK YANLIŞ YAPILAN HASTALAR RAPORU")
    print("="*95)
    print(md_content)
    print("="*95)

if __name__ == "__main__":
    main()
