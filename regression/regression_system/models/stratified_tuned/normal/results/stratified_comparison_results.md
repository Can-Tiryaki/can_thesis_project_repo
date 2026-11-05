# 📊 Tekil Tabakalı Bölme Model Sonuçları (Normal Veri - Tuned)
Bu tablo, temizlenmiş normal veri setinin (%80 Eğitim, %20 Test) olarak tabakalı şekilde tekil olarak bölünmesi ve modellerin **GridSearchCV** ile eğitilmesiyle elde edilen test seti başarılarını göstermektedir.

| Model | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Eğitim MSE | Test MSE | Test MAE | Test RMSE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.6568 | 0.5512 | 0.1056 | 0.2248 | 0.2364 | 0.3557 | 0.4862 | 🟢 Dengeli / Stabil Model |
| Gradient_Boosting | 0.7292 | 0.5280 | 0.2013 | 0.1774 | 0.2487 | 0.3519 | 0.4987 | 🟡 Hafif Overfitting |
| **XGBoost** | **0.7710** | **0.5747** | **0.1962** | **0.1500** | **0.2241** | **0.3481** | **0.4733** | **🟡 Hafif Overfitting** |
| Artificial_Neural_Network | 0.6346 | 0.5431 | 0.0915 | 0.2393 | 0.2407 | 0.3684 | 0.4906 | 🟢 Dengeli / Stabil Model |
| AdaBoost | 0.7535 | 0.5126 | 0.2409 | 0.1615 | 0.2568 | 0.3682 | 0.5067 | 🟡 Hafif Overfitting |
| Decision_Tree | 0.5978 | 0.4600 | 0.1378 | 0.2634 | 0.2845 | 0.3892 | 0.5334 | 🟡 Hafif Overfitting |