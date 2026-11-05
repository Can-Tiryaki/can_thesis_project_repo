# 📊 Tekil Tabakalı Bölme Master Karşılaştırma Sonuçları (Stratified Split Master Table)
Bu tablo; train-test bölünmesi sırasında **StratifiedShuffleSplit (Tabakalı Tekil Bölme)** kullanıldığında elde edilen train/test sonuçlarını karşılaştırmaktadır.

| Model | Deney Senaryosu | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Test MSE | Durum Analizi |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 1. Temel Veri (Varsayılan - Tabakalı) | 0.9427 | 0.5142 | 0.4285 | 0.2559 | ⚠️ Yüksek Overfitting |
| Gradient_Boosting | 1. Temel Veri (Varsayılan - Tabakalı) | 0.8610 | 0.4852 | 0.3757 | 0.2712 | ⚠️ Yüksek Overfitting |
| XGBoost | 1. Temel Veri (Varsayılan - Tabakalı) | 0.9995 | 0.3802 | 0.6193 | 0.3265 | ⚠️ Yüksek Overfitting |
| Artificial_Neural_Network | 1. Temel Veri (Varsayılan - Tabakalı) | 0.7106 | 0.4971 | 0.2134 | 0.2649 | 🟡 Hafif Overfitting |
| **AdaBoost** | **1. Temel Veri (Varsayılan - Tabakalı)** | **0.7351** | **0.5149** | **0.2201** | **0.2556** | **🟡 Hafif Overfitting** |
| Decision_Tree | 1. Temel Veri (Varsayılan - Tabakalı) | 1.0000 | 0.1114 | 0.8886 | 0.4682 | ⚠️ Yüksek Overfitting |
| Random_Forest | 2. Gelişmiş Veri (Varsayılan - Tabakalı) | 0.9422 | 0.5449 | 0.3973 | 0.2398 | ⚠️ Yüksek Overfitting |
| Gradient_Boosting | 2. Gelişmiş Veri (Varsayılan - Tabakalı) | 0.8708 | 0.4605 | 0.4102 | 0.2842 | ⚠️ Yüksek Overfitting |
| XGBoost | 2. Gelişmiş Veri (Varsayılan - Tabakalı) | 0.9998 | 0.4322 | 0.5677 | 0.2992 | ⚠️ Yüksek Overfitting |
| **Artificial_Neural_Network** | **2. Gelişmiş Veri (Varsayılan - Tabakalı)** | **0.7654** | **0.5457** | **0.2196** | **0.2393** | **🟡 Hafif Overfitting** |
| AdaBoost | 2. Gelişmiş Veri (Varsayılan - Tabakalı) | 0.7528 | 0.5211 | 0.2317 | 0.2523 | 🟡 Hafif Overfitting |
| Decision_Tree | 2. Gelişmiş Veri (Varsayılan - Tabakalı) | 1.0000 | 0.0965 | 0.9035 | 0.4760 | ⚠️ Yüksek Overfitting |
| Random_Forest | 3. Temel Veri (Tuned - Tabakalı) | 0.6568 | 0.5512 | 0.1056 | 0.2364 | 🟢 Dengeli / Stabil Model |
| Gradient_Boosting | 3. Temel Veri (Tuned - Tabakalı) | 0.7292 | 0.5280 | 0.2013 | 0.2487 | 🟡 Hafif Overfitting |
| **XGBoost** | **3. Temel Veri (Tuned - Tabakalı)** | **0.7710** | **0.5747** | **0.1962** | **0.2241** | **🟡 Hafif Overfitting** |
| Artificial_Neural_Network | 3. Temel Veri (Tuned - Tabakalı) | 0.6346 | 0.5431 | 0.0915 | 0.2407 | 🟢 Dengeli / Stabil Model |
| AdaBoost | 3. Temel Veri (Tuned - Tabakalı) | 0.7535 | 0.5126 | 0.2409 | 0.2568 | 🟡 Hafif Overfitting |
| Decision_Tree | 3. Temel Veri (Tuned - Tabakalı) | 0.5978 | 0.4600 | 0.1378 | 0.2845 | 🟡 Hafif Overfitting |
| **Random_Forest** | **4. Gelişmiş Veri (Tuned - Tabakalı)** | **0.6581** | **0.5635** | **0.0946** | **0.2300** | **🟢 Dengeli / Stabil Model** |
| Gradient_Boosting | 4. Gelişmiş Veri (Tuned - Tabakalı) | 0.7374 | 0.5356 | 0.2019 | 0.2447 | 🟡 Hafif Overfitting |
| XGBoost | 4. Gelişmiş Veri (Tuned - Tabakalı) | 0.7778 | 0.5419 | 0.2359 | 0.2413 | 🟡 Hafif Overfitting |
| Artificial_Neural_Network | 4. Gelişmiş Veri (Tuned - Tabakalı) | 0.6488 | 0.5539 | 0.0949 | 0.2350 | 🟢 Dengeli / Stabil Model |
| AdaBoost | 4. Gelişmiş Veri (Tuned - Tabakalı) | 0.7957 | 0.5225 | 0.2732 | 0.2516 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 4. Gelişmiş Veri (Tuned - Tabakalı) | 0.6065 | 0.5157 | 0.0908 | 0.2551 | 🟢 Dengeli / Stabil Model |