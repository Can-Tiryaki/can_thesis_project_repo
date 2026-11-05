import os
import shutil
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# api_model dizinini Python yoluna ekleyelim (Predictor'ı doğrudan yükleyebilmek için)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../api_model')))
# pyrefly: ignore [missing-import]
from predictor import HbA1cPredictor, ValidationError

app = Flask(__name__, template_folder='templates', static_folder='static')
PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")

def copy_assets():
    """Analiz esnasında üretilen grafikleri ve model sonuç resimlerini static klasörüne kopyalar."""
    base_path = PROJECT_ROOT
    dst_dir = os.path.join(os.path.dirname(__file__), "static/imgs")
    os.makedirs(dst_dir, exist_ok=True)
    
    # 1. Genel Analiz Görselleri (7 Adet)
    src_dir = os.path.join(base_path, "analysis/imgs")
    if os.path.exists(src_dir):
        for file_name in os.listdir(src_dir):
            if file_name.endswith('.png'):
                shutil.copy(os.path.join(src_dir, file_name), os.path.join(dst_dir, file_name))
                
    # 2. Tuned Model Görselleri
    tuned_results_dir = os.path.join(base_path, "models/tabakali_tuned/feature_engineered/results")
    if os.path.exists(tuned_results_dir):
        for src_name, dst_name in [
            ("train_test_comparison.png", "ml_train_test_comparison.png"),
            ("best_model_predictions.png", "ml_best_model_predictions.png"),
            ("best_model_feature_importance.png", "ml_best_model_feature_importance.png")
        ]:
            src_file = os.path.join(tuned_results_dir, src_name)
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(dst_dir, dst_name))

    # 3. Ağırlıklı Model Görselleri
    weighted_results_dir = os.path.join(base_path, "models/tabakali_weighted/feature_engineered/results")
    if os.path.exists(weighted_results_dir):
        for src_name, dst_name in [
            ("train_test_comparison.png", "ml_weighted_train_test_comparison.png"),
            ("best_model_predictions.png", "ml_weighted_best_model_predictions.png"),
            ("best_model_feature_importance.png", "ml_weighted_best_model_feature_importance.png")
        ]:
            src_file = os.path.join(weighted_results_dir, src_name)
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(dst_dir, dst_name))
        
    print("[*] Tüm genel analiz ve makine öğrenmesi grafik görselleri static/imgs klasörüne kopyalandı.")

# Başlangıçta grafikleri güncelle
copy_assets()

# Predictor nesnesini başlat
try:
    predictor = HbA1cPredictor()
    print("[*] SOLID Tahmin Motoru başarıyla yüklendi.")
except Exception as e:
    print(f"[!] Hata: Tahmin motoru yüklenemedi: {e}")
    predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Sunucuda model yüklenemedi.'}), 500
        
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON verisi bulunamadı.'}), 400
        
    try:
        # HbA1c Tahmini yap
        predicted_hba1c = predictor.predict_hba1c(data)
        
        # Risk sınıfı ve klinik açıklama
        if predicted_hba1c < 5.7:
            risk = 'Normal'
            risk_color = '#10b981' # Emerald Green
            desc = 'Tebrikler! HbA1c seviyeniz normal sınırlar içerisindedir. Sağlıklı beslenme ve aktif yaşam tarzını korumaya devam etmeniz önerilir.'
        elif predicted_hba1c < 6.5:
            risk = 'Prediyabet (Gizli Şeker)'
            risk_color = '#f59e0b' # Amber/Yellow
            desc = 'Dikkat! HbA1c seviyeniz prediyabet (gizli şeker) sınırındadır. Tip 2 diyabet riskini azaltmak için karbonhidrat tüketimini sınırlamanız ve fiziksel aktivitenizi arttırmanız tavsiye edilir.'
        else:
            risk = 'Diyabet (Şeker Hastalığı)'
            risk_color = '#ef4444' # Rose Red
            desc = 'HbA1c seviyeniz diyabet teşhis sınırındadır. Kesin teşhis, takip ve tedavi planlaması için bir Endokrinoloji veya İç Hastalıkları (Dahiliye) uzmanına başvurmanız önem arz etmektedir.'
            
        return jsonify({
            'predicted_hba1c': round(predicted_hba1c, 2),
            'risk_status': risk,
            'risk_color': risk_color,
            'description': desc
        })
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Tahmin işlemi sırasında sistemsel bir hata oluştu: {str(e)}'}), 500

if __name__ == '__main__':
    # Flask sunucusunu 5000 portunda başlat
    app.run(host='0.0.0.0', port=5000, debug=True)
