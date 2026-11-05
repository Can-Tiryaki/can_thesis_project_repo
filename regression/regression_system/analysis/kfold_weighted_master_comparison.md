# 📊 K-Fold Çapraz Doğrulama Ağırlıklı Bölme Master Karşılaştırma Sonuçları
Bu tablo; Nested 5-Fold Cross Validation bölünmesi sırasında diyabetik hastalara 5 kat, prediyabetiklere 2 kat ceza ağırlığı verildiğinde elde edilen ortalama train/validation sonuçlarını karşılaştırmaktadır.

| Model | Deney Senaryosu | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Doğrulama MSE | Diyabetik MAE | Durum Analizi |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 5. Temel Veri (Weighted - K-Fold) | 0.6819 | 0.4831 | 0.1987 | 0.3139 | 0.8784 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 5. Temel Veri (Weighted - K-Fold) | 0.7149 | 0.4398 | 0.2752 | 0.3391 | 0.9087 | ⚠️ Yüksek Overfitting |
| **XGBoost** | **5. Temel Veri (Weighted - K-Fold)** | **0.7769** | **0.5256** | **0.2512** | **0.2817** | **0.8425** | **⚠️ Yüksek Overfitting** |
| Artificial_Neural_Network | 5. Temel Veri (Weighted - K-Fold) | 0.6257 | 0.4361 | 0.1896 | 0.3661 | 0.9956 | 🟡 Hafif Overfitting |
| AdaBoost | 5. Temel Veri (Weighted - K-Fold) | 0.7552 | 0.4584 | 0.2968 | 0.3283 | 0.8131 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 5. Temel Veri (Weighted - K-Fold) | 0.6091 | 0.3285 | 0.2807 | 0.3968 | 0.9651 | ⚠️ Yüksek Overfitting |
| Random_Forest | 6. Gelişmiş Veri (Weighted - K-Fold) | 0.7035 | 0.4890 | 0.2145 | 0.3107 | 0.8838 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 6. Gelişmiş Veri (Weighted - K-Fold) | 0.7547 | 0.4704 | 0.2843 | 0.3179 | 0.8675 | ⚠️ Yüksek Overfitting |
| XGBoost | 6. Gelişmiş Veri (Weighted - K-Fold) | 0.7824 | 0.4970 | 0.2854 | 0.2972 | 0.8549 | ⚠️ Yüksek Overfitting |
| Artificial_Neural_Network | 6. Gelişmiş Veri (Weighted - K-Fold) | 0.6449 | 0.4082 | 0.2368 | 0.3891 | 0.9912 | 🟡 Hafif Overfitting |
| **AdaBoost** | **6. Gelişmiş Veri (Weighted - K-Fold)** | **0.7736** | **0.5370** | **0.2366** | **0.2749** | **0.8207** | **🟡 Hafif Overfitting** |
| Decision_Tree | 6. Gelişmiş Veri (Weighted - K-Fold) | 0.6446 | 0.2928 | 0.3518 | 0.4146 | 1.0019 | ⚠️ Yüksek Overfitting |