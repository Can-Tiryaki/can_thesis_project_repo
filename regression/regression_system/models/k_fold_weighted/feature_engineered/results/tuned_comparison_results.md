# 📊 Hiperparametre Ayarlı ve Ağırlıklı Model Sonuçları (Gelişmiş Veri)
Bu tablo, zenginleştirilmiş gelişmiş veri seti üzerinde **diyabetik hastalara 5 kat, prediyabetiklere 2 kat ceza ağırlığı uygulanarak** Nested 5-Fold Cross Validation ile elde edilen sonuçları göstermektedir.

| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Ortalama MAE | Diyabetik MAE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.7035 | 0.4890 | 0.2145 | 0.1862 | 0.3107 | 0.3730 | 0.8838 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 0.7547 | 0.4704 | 0.2843 | 0.1544 | 0.3179 | 0.3863 | 0.8675 | ⚠️ Yüksek Overfitting |
| XGBoost | 0.7824 | 0.4970 | 0.2854 | 0.1369 | 0.2972 | 0.3733 | 0.8549 | ⚠️ Yüksek Overfitting |
| Artificial_Neural_Network | 0.6449 | 0.4082 | 0.2368 | 0.2228 | 0.3891 | 0.3638 | 0.9912 | 🟡 Hafif Overfitting |
| **AdaBoost** | **0.7736** | **0.5370** | **0.2366** | **0.1420** | **0.2749** | **0.3657** | **0.8207** | **🟡 Hafif Overfitting** |
| Decision_Tree | 0.6446 | 0.2928 | 0.3518 | 0.2227 | 0.4146 | 0.4256 | 1.0019 | ⚠️ Yüksek Overfitting |