# 📊 K-Fold Çapraz Doğrulama Model Karşılaştırma Sonuçları (Gelişmiş Veri)
Bu tablo, özellik mühendisliği yapılmış zenginleştirilmiş veri seti (`mergeAllData_engineered.csv`) üzerinde 5-Fold Çapraz Doğrulama sonuçlarını karşılaştırmaktadır.

| Model | Ortalama R² | R² Std | Ortalama MSE | MSE Std | Ortalama MAE | MAE Std | Ortalama RMSE | RMSE Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random_Forest | 0.5014 | 0.1511 | 0.2969 | 0.0497 | 0.3719 | 0.0375 | 0.5429 | 0.0460 |
| Gradient_Boosting | 0.5093 | 0.1733 | 0.2883 | 0.0607 | 0.3690 | 0.0387 | 0.5338 | 0.0577 |
| XGBoost | 0.5271 | 0.1375 | 0.2801 | 0.0404 | 0.3691 | 0.0325 | 0.5278 | 0.0390 |
| Artificial_Neural_Network | -0.2103 | 1.0845 | 0.8529 | 0.9498 | 0.4430 | 0.0381 | 0.8234 | 0.4183 |
| **AdaBoost** | **0.5332** | **0.1499** | **0.2756** | **0.0543** | **0.3659** | **0.0275** | **0.5222** | **0.0534** |
| Decision_Tree | 0.3087 | 0.2302 | 0.4088 | 0.0804 | 0.4191 | 0.0403 | 0.6363 | 0.0628 |