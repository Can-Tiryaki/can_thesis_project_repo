# 📊 Hiperparametre Ayarlı (Tuned) Model Karşılaştırma Sonuçları (Gelişmiş Veri)
Bu tablo, özellik mühendisliği yapılmış zenginleştirilmiş veri seti (`mergeAllData_engineered.csv`) üzerinde **GridSearchCV + Nested 5-Fold Cross Validation** kullanılarak optimize edilen modellerin sonuçlarını göstermektedir.

| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Ortalama MAE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.7044 | 0.4964 | 0.2080 | 0.1861 | 0.3061 | 0.3686 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 0.7118 | 0.4893 | 0.2225 | 0.1807 | 0.3078 | 0.3692 | 🟡 Hafif Overfitting |
| **XGBoost** | **0.7502** | **0.5498** | **0.2004** | **0.1566** | **0.2693** | **0.3631** | **🟡 Hafif Overfitting** |
| Artificial_Neural_Network | 0.6449 | 0.4082 | 0.2368 | 0.2228 | 0.3891 | 0.3638 | 🟡 Hafif Overfitting |
| AdaBoost | 0.7440 | 0.5147 | 0.2292 | 0.1613 | 0.2856 | 0.3708 | 🟡 Hafif Overfitting |
| Decision_Tree | 0.6439 | 0.3236 | 0.3204 | 0.2233 | 0.3993 | 0.4105 | ⚠️ Yüksek Overfitting |