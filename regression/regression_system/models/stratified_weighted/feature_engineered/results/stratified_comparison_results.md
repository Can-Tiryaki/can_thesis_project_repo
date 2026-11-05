# 📊 Tekil Tabakalı Ağırlıklı Bölme Model Sonuçları (Gelişmiş Veri)
Bu tablo, özellik mühendisliği yapılmış zenginleştirilmiş veri setinin (%80 Eğitim, %20 Test) olarak tabakalı bölünmesi ve **diyabetik hastalara 5 kat, prediyabetiklere 2 kat daha fazla hata cezası (sample weight) verilerek** GridSearchCV ile eğitilmesiyle elde edilen sonuçları göstermektedir.

| Model | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Test MSE | Test MAE | Diyabetik Test MAE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.7037 | 0.5413 | 0.1623 | 0.2416 | 0.3570 | 0.4913 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 0.8202 | 0.4769 | 0.3433 | 0.2756 | 0.3728 | 0.6835 | ⚠️ Yüksek Overfitting |
| XGBoost | 0.8159 | 0.5140 | 0.3019 | 0.2560 | 0.3603 | 0.6535 | ⚠️ Yüksek Overfitting |
| **Artificial_Neural_Network** | **0.6488** | **0.5539** | **0.0949** | **0.2350** | **0.3553** | **0.7267** | **🟢 Dengeli / Stabil Model** |
| AdaBoost | 0.7853 | 0.5083 | 0.2770 | 0.2591 | 0.3590 | 0.6442 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 0.6367 | 0.4009 | 0.2359 | 0.3156 | 0.4001 | 0.6332 | 🟡 Hafif Overfitting |