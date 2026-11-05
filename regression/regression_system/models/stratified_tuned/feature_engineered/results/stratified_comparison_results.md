# 📊 Tekil Tabakalı Bölme Model Sonuçları (Gelişmiş Veri - Tuned)
Bu tablo, özellik mühendisliği yapılmış zenginleştirilmiş veri setinin (%80 Eğitim, %20 Test) olarak tabakalı şekilde tekil olarak bölünmesi ve modellerin **GridSearchCV** ile eğitilmesiyle elde edilen test seti başarılarını göstermektedir.

| Model | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Eğitim MSE | Test MSE | Test MAE | Test RMSE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random_Forest** | **0.6581** | **0.5635** | **0.0946** | **0.2240** | **0.2300** | **0.3515** | **0.4796** | **🟢 Dengeli / Stabil Model** |
| Gradient_Boosting | 0.7374 | 0.5356 | 0.2019 | 0.1720 | 0.2447 | 0.3506 | 0.4947 | 🟡 Hafif Overfitting |
| XGBoost | 0.7778 | 0.5419 | 0.2359 | 0.1456 | 0.2413 | 0.3581 | 0.4913 | 🟡 Hafif Overfitting |
| Artificial_Neural_Network | 0.6488 | 0.5539 | 0.0949 | 0.2301 | 0.2350 | 0.3553 | 0.4848 | 🟢 Dengeli / Stabil Model |
| AdaBoost | 0.7957 | 0.5225 | 0.2732 | 0.1339 | 0.2516 | 0.3639 | 0.5016 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 0.6065 | 0.5157 | 0.0908 | 0.2578 | 0.2551 | 0.3753 | 0.5051 | 🟢 Dengeli / Stabil Model |