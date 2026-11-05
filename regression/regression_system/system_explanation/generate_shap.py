import os
from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

def main():
    # Matplotlib backend set
    plt.switch_backend('Agg')
    
    base_dir = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")
    model_path = os.path.join(base_dir, "api_model/models/best_model.pkl")
    csv_path = os.path.join(base_dir, "feature_engineering/mergeAllData_engineered.csv")
    save_path = os.path.join(base_dir, "analysis/imgs/shap_summary.png")
    
    if not os.path.exists(model_path):
        print(f"Hata: Model bulunamadı: {model_path}")
        return
    if not os.path.exists(csv_path):
        print(f"Hata: Veri seti bulunamadı: {csv_path}")
        return
        
    print("[*] Model ve veri seti yükleniyor...")
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    
    # Türkçe sütun isimlerini snake_case eşleştirelim
    column_mapping = {
        'Yaş': 'yas',
        'Cinsiyet': 'cinsiyet',
        'Açlık Kan Şekeri (AKŞ)': 'aks',
        'Tokluk Kan Şekeri (TKŞ) 1.Saat': 'tks_1',
        'Tokluk Kan Şekeri (TKŞ) 2.Saat': 'tks_2',
        'Kolesterol (Total)': 'kolesterol',
        'Kolestrol (HDL)': 'hdl',
        'Kolestrol (LDL)': 'ldl',
        'Trigliserid': 'tg',
        'TG_HDL_Orani': 'tg_hdl_orani',
        'LDL_HDL_Orani': 'ldl_hdl_orani',
        'AKS_TKS_Orani': 'aks_tks_orani',
        'Glisemik_Ortalama': 'glisemik_ortalama'
    }
    
    feature_cols = list(column_mapping.keys())
    X = df[feature_cols].copy()
    X = X.rename(columns=column_mapping)
    
    print("[*] SHAP Değerleri hesaplanıyor (TreeExplainer)...")
    # TreeExplainer ile SHAP değerlerini hesaplayalım
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    print("[*] SHAP Beeswarm grafiği çiziliyor...")
    plt.figure(figsize=(11, 7))
    
    # SHAP summary plot (Beeswarm)
    shap.summary_plot(shap_values, X, show=False)
    
    plt.title("SHAP Özellik Etki Değerlendirme Analizi (Beeswarm Raporu)", fontsize=13, fontweight='bold', pad=20)
    
    # Eksen etiketlerini ve görünümü güzelleştir
    plt.tight_layout()
    
    # Kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] SHAP analizi görseli başarıyla oluşturuldu ve kaydedildi: {save_path}")

if __name__ == "__main__":
    main()
