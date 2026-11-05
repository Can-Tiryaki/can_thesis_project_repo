# 📊 Hiperparametre Ayarlı (Tuned) Model Karşılaştırma Sonuçları (Normal Veri)
Bu tablo, temizlenmiş normal veri seti (`mergeAllData_cleaned.csv`) üzerinde **GridSearchCV + Nested 5-Fold Cross Validation** kullanılarak optimize edilen modellerin sonuçlarını göstermektedir.

| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Ortalama MAE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.6904 | 0.4849 | 0.2055 | 0.1944 | 0.3147 | 0.3683 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 0.7181 | 0.4916 | 0.2265 | 0.1772 | 0.3046 | 0.3705 | 🟡 Hafif Overfitting |
| **XGBoost** | **0.7343** | **0.5493** | **0.1849** | **0.1664** | **0.2689** | **0.3588** | **🟡 Hafif Overfitting** |
| Artificial_Neural_Network | 0.6257 | 0.4361 | 0.1896 | 0.2350 | 0.3661 | 0.3604 | 🟡 Hafif Overfitting |
| AdaBoost | 0.7532 | 0.5024 | 0.2508 | 0.1549 | 0.2947 | 0.3699 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 0.6259 | 0.3324 | 0.2935 | 0.2353 | 0.3929 | 0.4104 | ⚠️ Yüksek Overfitting |