# 📊 Tekil Tabakalı Bölme Model Sonuçları (Normal Veri - Varsayılan)
Bu tablo, temizlenmiş normal veri setinin (%80 Eğitim, %20 Test) olarak tabakalı şekilde tekil olarak bölünmesi ve modellerin **varsayılan (default)** parametrelerle eğitilmesiyle elde edilen test seti başarılarını göstermektedir.

| Model | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Eğitim MSE | Test MSE | Test MAE | Test RMSE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.9427 | 0.5142 | 0.4285 | 0.0375 | 0.2559 | 0.3735 | 0.5059 | ⚠️ Yüksek Overfitting |
| Gradient_Boosting | 0.8610 | 0.4852 | 0.3757 | 0.0911 | 0.2712 | 0.3676 | 0.5208 | ⚠️ Yüksek Overfitting |
| XGBoost | 0.9995 | 0.3802 | 0.6193 | 0.0003 | 0.3265 | 0.4171 | 0.5714 | ⚠️ Yüksek Overfitting |
| Artificial_Neural_Network | 0.7106 | 0.4971 | 0.2134 | 0.1896 | 0.2649 | 0.3777 | 0.5147 | 🟡 Hafif Overfitting |
| **AdaBoost** | **0.7351** | **0.5149** | **0.2201** | **0.1736** | **0.2556** | **0.3696** | **0.5055** | **🟡 Hafif Overfitting** |
| Decision_Tree | 1.0000 | 0.1114 | 0.8886 | 0.0000 | 0.4682 | 0.5083 | 0.6842 | ⚠️ Yüksek Overfitting |