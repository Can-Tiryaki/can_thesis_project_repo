# 📊 Tekil Tabakalı Bölme Model Sonuçları (Gelişmiş Veri - Varsayılan)
Bu tablo, zenginleştirilmiş gelişmiş veri setinin (%80 Eğitim, %20 Test) olarak tabakalı şekilde tekil olarak bölünmesi ve modellerin **varsayılan (default)** parametrelerle eğitilmesiyle elde edilen test seti başarılarını göstermektedir.

| Model | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Eğitim MSE | Test MSE | Test MAE | Test RMSE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.9422 | 0.5449 | 0.3973 | 0.0379 | 0.2398 | 0.3616 | 0.4897 | ⚠️ Yüksek Overfitting |
| Gradient_Boosting | 0.8708 | 0.4605 | 0.4102 | 0.0847 | 0.2842 | 0.3789 | 0.5331 | ⚠️ Yüksek Overfitting |
| XGBoost | 0.9998 | 0.4322 | 0.5677 | 0.0001 | 0.2992 | 0.3997 | 0.5470 | ⚠️ Yüksek Overfitting |
| **Artificial_Neural_Network** | **0.7654** | **0.5457** | **0.2196** | **0.1537** | **0.2393** | **0.3736** | **0.4892** | **🟡 Hafif Overfitting** |
| AdaBoost | 0.7528 | 0.5211 | 0.2317 | 0.1619 | 0.2523 | 0.3615 | 0.5023 | 🟡 Hafif Overfitting |
| Decision_Tree | 1.0000 | 0.0965 | 0.9035 | 0.0000 | 0.4760 | 0.5171 | 0.6899 | ⚠️ Yüksek Overfitting |