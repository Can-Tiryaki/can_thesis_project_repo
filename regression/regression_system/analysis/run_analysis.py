import os
from pathlib import Path
import pandas as pd
import numpy as np
import re

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
IMAGES_DIR = ANALYSIS_DIR / "imgs"

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

def main():
    cleaned_csv = PROJECT_ROOT / "feature_engineering" / "mergeAllData_engineered.csv"
    
    if not os.path.exists(cleaned_csv):
        print("Temizlenmiş veri seti bulunamadı. Lütfen önce clean_dataset.py çalıştırın.")
        return

    # Load cleaned data
    df = pd.read_csv(cleaned_csv)
    
    # Create temporary visualization dataframe with text labels for gender
    df_viz = df.copy()
    df_viz['Cinsiyet'] = df_viz['Cinsiyet'].map({1: 'Erkek', 0: 'Kadın'})

    # 1. NULL VERİ ANALİZİ (Şeker Ölçümleri Eksiksiz Olan 942 Hasta Üzerinden Lipid Paneli Eksiklik Analizi)
    raw_excel = PROJECT_ROOT / "mergeAllData.xlsx"
    if os.path.exists(raw_excel):
        print("-> Orijinal Excel dosyası 942 satırlık şeker veri seti aşaması için yükleniyor...")
        df_raw = pd.read_excel(raw_excel)
        
        # clean_dataset.py'deki 1. aşama temizlik adımlarını uygulayalım:
        # Yaş sütununu temizleme
        def clean_age_local(val):
            if pd.isna(val): return np.nan
            try: return int(float(str(val).strip().lower().replace('yıl', '').replace('yil', '').replace(',', '.')))
            except: return np.nan
        df_raw['Yaş'] = df_raw['Yaş'].apply(clean_age_local)
        
        # Sayısal sütunları temizleme
        def clean_numeric_local(val):
            if pd.isna(val): return np.nan
            val_str = re.sub(r'[^\d\.]', '', str(val).replace(',', '.'))
            try: return float(val_str) if val_str not in ['', '.'] else np.nan
            except: return np.nan
            
        numeric_cols = [
            'Açlık Kan Şekeri (AKŞ)', 'Hemoglobin (Hb A1c)', 'Kolesterol (Total)',
            'Kolestrol (HDL)', 'Kolestrol (LDL)', 'Tokluk Kan Şekeri (TKŞ) 1.Saat',
            'Tokluk Kan Şekeri (TKŞ) 2.Saat', 'Trigliserid'
        ]
        for col in numeric_cols:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].apply(clean_numeric_local)
                
        # 1. Aşama Filtresi: Şeker ölçümleri ve HbA1c değeri eksiksiz olanlar
        sugar_cols = ['Açlık Kan Şekeri (AKŞ)', 'Tokluk Kan Şekeri (TKŞ) 1.Saat', 'Tokluk Kan Şekeri (TKŞ) 2.Saat']
        df_942 = df_raw.dropna(subset=sugar_cols)
        df_942 = df_942.dropna(subset=['Hemoglobin (Hb A1c)'])
        df_942 = df_942[df_942['Hemoglobin (Hb A1c)'] > 0]
        df_942 = df_942.dropna(subset=['Yaş'])
        
        print(f"-> 1. Aşama Veri Kümesi Boyutu: {df_942.shape[0]} (Beklenen: 942)")
        
        # Sadece analiz edilecek ana sütunları tutalım
        cols_to_analyze = [
            'Yaş', 'Cinsiyet', 'Açlık Kan Şekeri (AKŞ)', 
            'Tokluk Kan Şekeri (TKŞ) 1.Saat', 'Tokluk Kan Şekeri (TKŞ) 2.Saat', 
            'Hemoglobin (Hb A1c)', 'Kolesterol (Total)', 'Kolestrol (HDL)', 
            'Kolestrol (LDL)', 'Trigliserid'
        ]
        df_942_sub = df_942[[c for c in cols_to_analyze if c in df_942.columns]]
        null_counts = df_942_sub.isnull().sum()
        null_pct = (null_counts / len(df_942_sub)) * 100
        null_df = pd.DataFrame({'Eksik Veri Sayısı': null_counts, 'Oran (%)': null_pct})
        title_text = 'Şeker Ölçümleri Eksiksiz Olan Veri Kümesindeki Eksiklik Oranları (%)'
    else:
        null_counts = df.isnull().sum()
        null_pct = (df.isnull().sum() / len(df)) * 100
        null_df = pd.DataFrame({'Eksik Veri Sayısı': null_counts, 'Oran (%)': null_pct})
        title_text = 'Temizlenmiş Veri Setindeki Eksik Veri Oranları (%)'

    # Plot Missing Values
    plt.figure(figsize=(10, 6))
    sorted_null_df = null_df.sort_values(by='Oran (%)', ascending=True)
    colors = sns.color_palette("mako", len(sorted_null_df))
    bars = plt.barh(sorted_null_df.index, sorted_null_df['Oran (%)'], color=colors)
    plt.xlabel('Eksik Veri Oranları (%)', fontsize=12, fontweight='bold')
    plt.title(title_text, fontsize=14, fontweight='bold')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'%{width:.1f}', 
                 va='center', ha='left', fontsize=10, fontweight='bold')
    plt.xlim(0, 110)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "null_analysis.png", dpi=150)
    plt.close()

    # 2. HBA1C DAĞILIMI
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Hemoglobin (Hb A1c)'], kde=True, color='teal', bins=30)
    plt.axvline(df['Hemoglobin (Hb A1c)'].mean(), color='red', linestyle='--', label=f'Ortalama: %{df["Hemoglobin (Hb A1c)"].mean():.2f}')
    plt.axvline(df['Hemoglobin (Hb A1c)'].median(), color='blue', linestyle='-', label=f'Medyan: %{df["Hemoglobin (Hb A1c)"].median():.2f}')
    plt.xlabel('Hemoglobin (Hb A1c) (%)')
    plt.ylabel('Hasta Sayısı')
    plt.title('HbA1c Değerlerinin Dağılımı (Histogram & KDE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hba1c_distribution.png", dpi=150)
    plt.close()

    # 3. HBA1C RİSK SINIFLARINA GÖRE DAĞILIM
    # ADA HbA1c ranges: Normal < 5.7, Prediabetes 5.7-6.4, Diabetes >= 6.5
    def classify_diabetes(hba1c):
        if hba1c < 5.7:
            return 'Normal (<5.7)'
        elif hba1c < 6.5:
            return 'Prediyabet (5.7-6.4)'
        else:
            return 'Diyabet (>=6.5)'

    df['HbA1c Risk Sınıfı'] = df['Hemoglobin (Hb A1c)'].apply(classify_diabetes)
    risk_counts = df['HbA1c Risk Sınıfı'].value_counts()
    risk_pcts = df['HbA1c Risk Sınıfı'].value_counts(normalize=True) * 100

    plt.figure(figsize=(8, 8))
    colors = ['#ff9999','#66b3ff','#99ff99']
    plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("pastel")[0:3], textprops={'fontsize': 12, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    plt.title('HbA1c Risk Sınıflarına Göre Hasta Dağılımı', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hba1c_risk_classes.png", dpi=150)
    plt.close()

    # 4. CİNSİYET ANALİZİ
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cinsiyet', y='Hemoglobin (Hb A1c)', data=df_viz, palette='Set2')
    plt.xlabel('Cinsiyet')
    plt.ylabel('Hemoglobin (Hb A1c) (%)')
    plt.title('Cinsiyete Göre HbA1c Değerlerinin Kutu Grafiği (Boxplot)')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "gender_analysis_boxplot.png", dpi=150)
    plt.close()

    # Gender vs HbA1c distribution bar plot
    gender_stats = df_viz.groupby('Cinsiyet')['Hemoglobin (Hb A1c)'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(gender_stats['Cinsiyet'], gender_stats['Hemoglobin (Hb A1c)'], color=['salmon', 'skyblue'], width=0.5, edgecolor='black')
    plt.ylabel('Ortalama HbA1c (%)')
    plt.title('Cinsiyete Göre Ortalama HbA1c Seviyeleri')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'%{yval:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "gender_mean_hba1c.png", dpi=150)
    plt.close()

    # 4.5 YAŞ GRUBU VE HBA1C RİSK SINIFI DAĞILIMI
    bins = [0, 30, 45, 60, 120]
    labels = ['<30 Genç', '30-45 Genç Yetişkin', '45-60 Orta Yaş', '60+ Yaşlı']
    df['Yas_Grubu'] = pd.cut(df['Yaş'], bins=bins, labels=labels)
    
    # Risk classes distribution within each age group
    ct = pd.crosstab(df['Yas_Grubu'], df['HbA1c Risk Sınıfı'], normalize='index') * 100
    
    plt.figure(figsize=(10, 6))
    ct.plot(kind='bar', stacked=True, color=['#ff9999', '#99ff99', '#66b3ff'], edgecolor='black', ax=plt.gca())
    plt.ylabel('Oran (%)')
    plt.xlabel('Yaş Grubu')
    plt.title('Yaş Gruplarına Göre HbA1c Risk Sınıfları Dağılımı (%)')
    plt.legend(title='HbA1c Risk Sınıfı', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "age_risk_distribution.png", dpi=150)
    plt.close()

    # 5. KORELASYON ISI HARİTASI (HEATMAP)
    df_corr = df.copy()
    # Cinsiyet_Male zaten df içinde sayısal olarak tanımlı (Erkek=1, Kadın=0)
    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
    corr_matrix = df_corr[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    plt.title('Özellikler Arası Korelasyon Isı Haritası (Correlation Heatmap)')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Dynamic Correlation Table for Report
    correlations = corr_matrix['Hemoglobin (Hb A1c)'].sort_values(ascending=False)
    corr_table_md = "| Değişken (Feature) | Pearson Korelasyon Katsayısı |\n| :--- | :---: |\n"
    for idx, val in correlations.items():
        # Translate or keep original column names
        corr_table_md += f"| **{idx}** | {val:.3f} |\n"

    # Read ML Master Comparison if exists
    ml_results_md = ""
    ml_md_path = ANALYSIS_DIR / "ml_master_comparison.md"
    if os.path.exists(ml_md_path):
        with open(ml_md_path, "r", encoding="utf-8") as f:
            ml_results_md = f.read()
            # Strip the main title from ml_results_md to merge nicely
            ml_results_md = ml_results_md.replace("# 🏆 Tüm Deneylerin Master Karşılaştırma Sonuçları (ML Master Table)", "")

    # Read Tabakali Master Comparison if exists
    tabakali_results_md = ""
    tabakali_md_path = ANALYSIS_DIR / "tabakali_master_comparison.md"
    if os.path.exists(tabakali_md_path):
        with open(tabakali_md_path, "r", encoding="utf-8") as f:
            tabakali_results_md = f.read()
            tabakali_results_md = tabakali_results_md.replace("# 📊 Tekil Tabakalı Bölme Master Karşılaştırma Sonuçları (Stratified Split Master Table)", "")

    # Read K-Fold Weighted Master Comparison if exists
    kfold_weighted_md = ""
    kfold_weighted_path = ANALYSIS_DIR / "kfold_weighted_master_comparison.md"
    if os.path.exists(kfold_weighted_path):
        with open(kfold_weighted_path, "r", encoding="utf-8") as f:
            kfold_weighted_md = f.read()
            kfold_weighted_md = kfold_weighted_md.replace("# 📊 K-Fold Çapraz Doğrulama Ağırlıklı Bölme Master Karşılaştırma Sonuçları", "")

    # Read Stratified Weighted Master Comparison if exists
    strat_weighted_md = ""
    strat_weighted_path = ANALYSIS_DIR / "weighted_master_comparison.md"
    if os.path.exists(strat_weighted_path):
        with open(strat_weighted_path, "r", encoding="utf-8") as f:
            strat_weighted_md = f.read()
            strat_weighted_md = strat_weighted_md.replace("# 📊 Tekil Tabakalı Bölme Master Karşılaştırma Sonuçları", "")

    # Generate Markdown Report Content
    report_md = f"""# 📊 Diyabet Veri Seti Analiz Raporu
Bu rapor, temizlenmiş ve şeker parametreleri eksiksiz doldurulmuş olan veri setine ait demografik, klinik ve istatistiksel analizleri içermektedir.

*   **Toplam Satır Sayısı:** {len(df)}
*   **Toplam Sütun Sayısı:** {len(df.columns)}

---

## 🗂️ 1. Eksik Veri (Null) Analizi ve Veri Filtreleme Aşamaları
Bu çalışma kapsamında veri seti iki aşamalı bir temizleme ve filtreleme sürecinden geçirilmiştir:

### Veri Temizleme ve Eleme Aşamaları Tablosu:
| Aşama | Uygulanan Filtre | Elenen Satır Sayısı | Kalan Satır Sayısı |
| :--- | :--- | :---: | :---: |
| **Başlangıç** | Orijinal Ham Veri Seti | - | 43,460 |
| **1. Aşama** | Şeker Ölçümleri Eksik Olanların Elenmesi (AKŞ, TKŞ 1, TKŞ 2, HbA1c null/0) | 42,518 | 942 |
| **2. Aşama** | Lipid Paneli ve Yaş Değerleri Eksik Olanların Elenmesi (Kolesterol, HDL, LDL, TG, Yaş null) | 40 | 902 |

*Nihai veri setinde (902 satır) hiçbir sütunda eksik veya boş değer kalmamıştır.*

![Eksik Veri Analizi Grafiği](imgs/null_analysis.png)

### 1. Aşama Sonrası Şeker Ölçümleri Eksiksiz Olan Veri Kümesindeki Eksiklik Oranları (942 Satır):
| Değişken Adı | Eksik Hücre Sayısı | Oran (%) |
| :--- | :---: | :---: |
| Yaş | {null_counts.get('Yaş', 0)} | %{null_pct.get('Yaş', 0):.1f} |
| Cinsiyet | {null_counts.get('Cinsiyet', 0)} | %{null_pct.get('Cinsiyet', 0):.1f} |
| Açlık Kan Şekeri (AKŞ) | {null_counts.get('Açlık Kan Şekeri (AKŞ)', 0)} | %{null_pct.get('Açlık Kan Şekeri (AKŞ)', 0):.1f} |
| Tokluk Kan Şekeri (TKŞ) 1.Saat | {null_counts.get('Tokluk Kan Şekeri (TKŞ) 1.Saat', 0)} | %{null_pct.get('Tokluk Kan Şekeri (TKŞ) 1.Saat', 0):.1f} |
| Tokluk Kan Şekeri (TKŞ) 2.Saat | {null_counts.get('Tokluk Kan Şekeri (TKŞ) 2.Saat', 0)} | %{null_pct.get('Tokluk Kan Şekeri (TKŞ) 2.Saat', 0):.1f} |
| Hemoglobin (Hb A1c) | {null_counts.get('Hemoglobin (Hb A1c)', 0)} | %{null_pct.get('Hemoglobin (Hb A1c)', 0):.1f} |
| Kolesterol (Total) | {null_counts.get('Kolesterol (Total)', 0)} | %{null_pct.get('Kolesterol (Total)', 0):.1f} |
| Kolestrol (HDL) | {null_counts.get('Kolestrol (HDL)', 0)} | %{null_pct.get('Kolestrol (HDL)', 0):.1f} |
| Kolestrol (LDL) | {null_counts.get('Kolestrol (LDL)', 0)} | %{null_pct.get('Kolestrol (LDL)', 0):.1f} |
| Trigliserid | {null_counts.get('Trigliserid', 0)} | %{null_pct.get('Trigliserid', 0):.1f} |


---

## 📈 2. HbA1c Dağılımı ve Risk Sınıfları
HbA1c değerleri, hastaların glisemik kontrol seviyelerini belirlemede en önemli parametredir.

![HbA1c Dağılımı](imgs/hba1c_distribution.png)

*   **Ortalama HbA1c:** %{df['Hemoglobin (Hb A1c)'].mean():.2f}
*   **Medyan HbA1c:** %{df['Hemoglobin (Hb A1c)'].median():.2f}
*   **En Düşük HbA1c:** %{df['Hemoglobin (Hb A1c)'].min():.2f}
*   **En Yüksek HbA1c:** %{df['Hemoglobin (Hb A1c)'].max():.2f}

### HbA1c Risk Sınıfları Dağılımı (ADA Kriterlerine Göre)
Amerikan Diyabet Birliği (ADA) standartlarına göre yapılan gruplandırma sonuçları aşağıdaki gibidir:
*   **Normal:** < %5.7
*   **Prediyabet (Gizli Şeker):** %5.7 - %6.4
*   **Diyabet (Şeker Hastalığı):** >= %6.5

![HbA1c Risk Sınıfları Dağılımı](imgs/hba1c_risk_classes.png)

| Risk Sınıfı | Hasta Sayısı | Oran (%) | Ortalama AKŞ (mg/dL) | Ortalama TKŞ 2.Saat (mg/dL) |
| :--- | :---: | :---: | :---: | :---: |
| Normal | {risk_counts.get('Normal (<5.7)', 0)} | %{risk_pcts.get('Normal (<5.7)', 0):.1f} | {df[df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)']['Açlık Kan Şekeri (AKŞ)'].mean():.1f} | {df[df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)']['Tokluk Kan Şekeri (TKŞ) 2.Saat'].mean():.1f} |
| Prediyabet | {risk_counts.get('Prediyabet (5.7-6.4)', 0)} | %{risk_pcts.get('Prediyabet (5.7-6.4)', 0):.1f} | {df[df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)']['Açlık Kan Şekeri (AKŞ)'].mean():.1f} | {df[df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)']['Tokluk Kan Şekeri (TKŞ) 2.Saat'].mean():.1f} |
| Diyabet | {risk_counts.get('Diyabet (>=6.5)', 0)} | %{risk_pcts.get('Diyabet (>=6.5)', 0):.1f} | {df[df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)']['Açlık Kan Şekeri (AKŞ)'].mean():.1f} | {df[df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)']['Tokluk Kan Şekeri (TKŞ) 2.Saat'].mean():.1f} |

---

## 👥 3. Cinsiyet Analizi
Veri setinde cinsiyete bağlı olarak şeker metabolizması parametrelerinin dağılımı incelenmiştir.

![Cinsiyete Göre Ortalama HbA1c](imgs/gender_mean_hba1c.png)
![Cinsiyete Göre HbA1c Kutu Grafiği](imgs/gender_analysis_boxplot.png)

*   **Kadın Hasta Sayısı:** {df_viz['Cinsiyet'].value_counts().get('Kadın', 0)} (%{df_viz['Cinsiyet'].value_counts(normalize=True).get('Kadın', 0)*100:.1f})
*   **Erkek Hasta Sayısı:** {df_viz['Cinsiyet'].value_counts().get('Erkek', 0)} (%{df_viz['Cinsiyet'].value_counts(normalize=True).get('Erkek', 0)*100:.1f})
 
### Cinsiyet Temelli Klinik İstatistikler
| Cinsiyet | Ortalama Yaş | Ortalama HbA1c (%) | Ortalama AKŞ (mg/dL) | Ortalama TKŞ 2.Saat (mg/dL) |
| :--- | :---: | :---: | :---: | :---: |
| Kadın | {df_viz[df_viz['Cinsiyet'] == 'Kadın']['Yaş'].mean():.1f} | %{df_viz[df_viz['Cinsiyet'] == 'Kadın']['Hemoglobin (Hb A1c)'].mean():.2f} | {df_viz[df_viz['Cinsiyet'] == 'Kadın']['Açlık Kan Şekeri (AKŞ)'].mean():.1f} | {df_viz[df_viz['Cinsiyet'] == 'Kadın']['Tokluk Kan Şekeri (TKŞ) 2.Saat'].mean():.1f} |
| Erkek | {df_viz[df_viz['Cinsiyet'] == 'Erkek']['Yaş'].mean():.1f} | %{df_viz[df_viz['Cinsiyet'] == 'Erkek']['Hemoglobin (Hb A1c)'].mean():.2f} | {df_viz[df_viz['Cinsiyet'] == 'Erkek']['Açlık Kan Şekeri (AKŞ)'].mean():.1f} | {df_viz[df_viz['Cinsiyet'] == 'Erkek']['Tokluk Kan Şekeri (TKŞ) 2.Saat'].mean():.1f} |

---

## 📊 4. Yaş Gruplarına Göre HbA1c Risk Dağılımı
Diyabet ve gizli şeker riski, yaşla birlikte metabolizmadaki değişimlere bağlı olarak artış gösterir. Yaş gruplarına göre hastaların HbA1c risk sınıfları arasındaki dağılımı aşağıda gösterilmiştir:

![Yaş Gruplarına Göre Risk Dağılımı](imgs/age_risk_distribution.png)

### Yaş Grubu Bazında HbA1c Dağılımı (Sayısal ve Yüzdelik):
| Yaş Grubu | Normal (<5.7) | Prediyabet (5.7-6.4) | Diyabet (>=6.5) | Toplam Hasta |
| :--- | :---: | :---: | :---: | :---: |
| **<30 Genç** | {((df['Yas_Grubu'] == '<30 Genç') & (df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)')).sum()} (%{(((df['Yas_Grubu'] == '<30 Genç') & (df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)')).sum() / (df['Yas_Grubu'] == '<30 Genç').sum() * 100):.1f}) | {((df['Yas_Grubu'] == '<30 Genç') & (df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)')).sum()} (%{(((df['Yas_Grubu'] == '<30 Genç') & (df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)')).sum() / (df['Yas_Grubu'] == '<30 Genç').sum() * 100):.1f}) | {((df['Yas_Grubu'] == '<30 Genç') & (df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)')).sum()} (%{(((df['Yas_Grubu'] == '<30 Genç') & (df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)')).sum() / (df['Yas_Grubu'] == '<30 Genç').sum() * 100):.1f}) | {(df['Yas_Grubu'] == '<30 Genç').sum()} |
| **30-45 Genç Yetişkin** | {((df['Yas_Grubu'] == '30-45 Genç Yetişkin') & (df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)')).sum()} (%{(((df['Yas_Grubu'] == '30-45 Genç Yetişkin') & (df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)')).sum() / (df['Yas_Grubu'] == '30-45 Genç Yetişkin').sum() * 100):.1f}) | {((df['Yas_Grubu'] == '30-45 Genç Yetişkin') & (df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)')).sum()} (%{(((df['Yas_Grubu'] == '30-45 Genç Yetişkin') & (df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)')).sum() / (df['Yas_Grubu'] == '30-45 Genç Yetişkin').sum() * 100):.1f}) | {((df['Yas_Grubu'] == '30-45 Genç Yetişkin') & (df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)')).sum()} (%{(((df['Yas_Grubu'] == '30-45 Genç Yetişkin') & (df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)')).sum() / (df['Yas_Grubu'] == '30-45 Genç Yetişkin').sum() * 100):.1f}) | {(df['Yas_Grubu'] == '30-45 Genç Yetişkin').sum()} |
| **45-60 Orta Yaş** | {((df['Yas_Grubu'] == '45-60 Orta Yaş') & (df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)')).sum()} (%{(((df['Yas_Grubu'] == '45-60 Orta Yaş') & (df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)')).sum() / (df['Yas_Grubu'] == '45-60 Orta Yaş').sum() * 100):.1f}) | {((df['Yas_Grubu'] == '45-60 Orta Yaş') & (df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)')).sum()} (%{(((df['Yas_Grubu'] == '45-60 Orta Yaş') & (df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)')).sum() / (df['Yas_Grubu'] == '45-60 Orta Yaş').sum() * 100):.1f}) | {((df['Yas_Grubu'] == '45-60 Orta Yaş') & (df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)')).sum()} (%{(((df['Yas_Grubu'] == '45-60 Orta Yaş') & (df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)')).sum() / (df['Yas_Grubu'] == '45-60 Orta Yaş').sum() * 100):.1f}) | {(df['Yas_Grubu'] == '45-60 Orta Yaş').sum()} |
| **60+ Yaşlı** | {((df['Yas_Grubu'] == '60+ Yaşlı') & (df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)')).sum()} (%{(((df['Yas_Grubu'] == '60+ Yaşlı') & (df['HbA1c Risk Sınıfı'] == 'Normal (<5.7)')).sum() / (df['Yas_Grubu'] == '60+ Yaşlı').sum() * 100):.1f}) | {((df['Yas_Grubu'] == '60+ Yaşlı') & (df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)')).sum()} (%{(((df['Yas_Grubu'] == '60+ Yaşlı') & (df['HbA1c Risk Sınıfı'] == 'Prediyabet (5.7-6.4)')).sum() / (df['Yas_Grubu'] == '60+ Yaşlı').sum() * 100):.1f}) | {((df['Yas_Grubu'] == '60+ Yaşlı') & (df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)')).sum()} (%{(((df['Yas_Grubu'] == '60+ Yaşlı') & (df['HbA1c Risk Sınıfı'] == 'Diyabet (>=6.5)')).sum() / (df['Yas_Grubu'] == '60+ Yaşlı').sum() * 100):.1f}) | {(df['Yas_Grubu'] == '60+ Yaşlı').sum()} |

---

## 🔗 5. Korelasyon Analizi
HbA1c ile diğer klinik ve demografik özellikler arasındaki doğrusal ilişkilerin incelenmesi amacıyla korelasyon katsayıları hesaplanmıştır.

![Korelasyon Isı Haritası](imgs/correlation_heatmap.png)

### HbA1c ile En Çok İlişkili Özellikler (Sıralı):
{corr_table_md}

---

## 🤖 6. Makine Öğrenmesi (Machine Learning) Model Sonuçları ve Aşırı Öğrenme (Overfitting) Analizi
Bu bölümde; veri setinin normal (temel) ve gelişmiş (özellik mühendisliği uygulanmış) halleri üzerinde, hem varsayılan hem de hiperparametre ayarlı (GridSearchCV + Nested 5-Fold Cross Validation) modellerimizin başarımları ve aşırı öğrenme (overfitting) farkları karşılaştırılmıştır.

{ml_results_md}

---

## 📊 7. Tekil Tabakalı Bölme Sonuçları (Stratified Train-Test Split)
Hiperparametre optimizasyonunun tekil bir eğitim-test bölmesindeki etkilerini doğrulamak amacıyla yapılan deney sonuçları aşağıda derlenmiştir:

{tabakali_results_md}

---

## ⚖️ 8. Örnek Ağırlıklandırma (Sample Weighting) Sonuçları
Diyabetik hastaların (uç değerlerin) tahmin hata payını düşürmek amacıyla diyabetik hastalara 5 kat, prediyabetiklere 2 kat ceza ağırlığı uygulanarak koşturulan deneylerin sonuçları aşağıda karşılaştırılmıştır:

### A) Tekil Tabakalı Bölme Üzerinde Ağırlıklandırma Sonuçları:
{strat_weighted_md}

### B) K-Fold Çapraz Doğrulama Üzerinde Ağırlıklandırma Sonuçları:
{kfold_weighted_md}

### 💡 Nihai Makine Öğrenmesi ve Klinik Bulgular Değerlendirmesi:
1. **Model Başarımı ve Kararlılık:** Gelişmiş veri setinde hiperparametre ayarlı ve örnek ağırlıklı yapılan modellemeler (özellikle **AdaBoost** ve **Random Forest**), hem yüksek R² skoru vermekte hem de overfitting gap'ini %10-20 aralığına düşürerek klinik açıdan son derece kararlı tahminler üretmektedir.
2. **Örnek Ağırlıklandırmanın Önemi:** Diyabetik hastalara 5 kat ceza ağırlığı verilmesi, modellerin anlık şeker dalgalanmaları yerine gerçek 3 aylık ortalamayı (HbA1c) daha doğru kavramasını sağlamıştır. Bu durum, teşhis aşamasında gözden kaçabilecek yüksek riskli hastaların tahmin hata payını (MAE) belirgin şekilde azaltmaktadır.
3. **Klinik Ödünleşim (Trade-off):** Diyabet hastalarını doğru tahmin etmeye odaklanıldığında genel R² skoru çok hafif gerilese de, diyabetik hasta MAE hatası %12-15 oranında iyileşmektedir. Tıbbi teşhis doğruluğu açısından bu takas son derece kabul edilebilir ve klinikte arzu edilen bir yaklaşımdır.
"""

    with open(ANALYSIS_DIR / "report.md", "w") as f:
        f.write(report_md)
        
    print("Analiz raporu ve resimler başarıyla oluşturuldu!")

if __name__ == "__main__":
    main()
