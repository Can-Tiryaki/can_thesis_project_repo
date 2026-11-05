# 📊 Hiperparametre Ayarlı ve Ağırlıklı Model Sonuçları (Normal Veri)
Bu tablo, temizlenmiş normal veri seti üzerinde **diyabetik hastalara 5 kat, prediyabetiklere 2 kat ceza ağırlığı uygulanarak** Nested 5-Fold Cross Validation ile elde edilen sonuçları göstermektedir.

| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Ortalama MAE | Diyabetik MAE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.6819 | 0.4831 | 0.1987 | 0.1998 | 0.3139 | 0.3723 | 0.8784 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 0.7149 | 0.4398 | 0.2752 | 0.1778 | 0.3391 | 0.3927 | 0.9087 | ⚠️ Yüksek Overfitting |
| **XGBoost** | **0.7769** | **0.5256** | **0.2512** | **0.1403** | **0.2817** | **0.3729** | **0.8425** | **⚠️ Yüksek Overfitting** |
| Artificial_Neural_Network | 0.6257 | 0.4361 | 0.1896 | 0.2350 | 0.3661 | 0.3604 | 0.9956 | 🟡 Hafif Overfitting |
| AdaBoost | 0.7552 | 0.4584 | 0.2968 | 0.1541 | 0.3283 | 0.3756 | 0.8131 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 0.6091 | 0.3285 | 0.2807 | 0.2442 | 0.3968 | 0.4142 | 0.9651 | ⚠️ Yüksek Overfitting |