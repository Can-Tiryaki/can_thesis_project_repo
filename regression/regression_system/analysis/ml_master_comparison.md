# 🏆 Tüm Deneylerin Master Karşılaştırma Sonuçları (ML Master Table)
Bu tablo; ham veri, özellik mühendisliği yapılmış veri ve hiperparametre ayarlarının (tuning) model performansları ile overfitting durumları üzerindeki etkilerini bir arada göstermektedir.

| Model | Deney Senaryosu | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Doğrulama MSE | Durum Analizi |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 1. Temel Veri (Varsayılan) | 0.8686 | 0.4842 | 0.3844 | 0.3055 | ⚠️ Yüksek Overfitting |
| Gradient_Boosting | 1. Temel Veri (Varsayılan) | 0.9021 | 0.4837 | 0.4184 | 0.3038 | ⚠️ Yüksek Overfitting |
| **XGBoost** | **1. Temel Veri (Varsayılan)** | **0.8686** | **0.5339** | **0.3347** | **0.2778** | **⚠️ Yüksek Overfitting** |
| Artificial_Neural_Network | 1. Temel Veri (Varsayılan) | 0.7774 | 0.1430 | 0.6344 | 0.5809 | ⚠️ Yüksek Overfitting |
| AdaBoost | 1. Temel Veri (Varsayılan) | 0.7720 | 0.5000 | 0.2720 | 0.2999 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 1. Temel Veri (Varsayılan) | 0.6843 | 0.3379 | 0.3464 | 0.3914 | ⚠️ Yüksek Overfitting |
| Random_Forest | 2. Gelişmiş Veri (Varsayılan) | 0.8745 | 0.5014 | 0.3731 | 0.2969 | ⚠️ Yüksek Overfitting |
| Gradient_Boosting | 2. Gelişmiş Veri (Varsayılan) | 0.9139 | 0.5093 | 0.4045 | 0.2883 | ⚠️ Yüksek Overfitting |
| XGBoost | 2. Gelişmiş Veri (Varsayılan) | 0.8815 | 0.5271 | 0.3544 | 0.2801 | ⚠️ Yüksek Overfitting |
| Artificial_Neural_Network | 2. Gelişmiş Veri (Varsayılan) | 0.8091 | -0.2103 | 1.0194 | 0.8529 | ⚠️ Yüksek Overfitting |
| **AdaBoost** | **2. Gelişmiş Veri (Varsayılan)** | **0.7807** | **0.5332** | **0.2476** | **0.2756** | **🟡 Hafif Overfitting** |
| Decision_Tree | 2. Gelişmiş Veri (Varsayılan) | 0.6895 | 0.3087 | 0.3808 | 0.4088 | ⚠️ Yüksek Overfitting |
| Random_Forest | 3. Temel Veri (Hiperparametre Ayarlı) | 0.6904 | 0.4849 | 0.2055 | 0.3147 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 3. Temel Veri (Hiperparametre Ayarlı) | 0.7181 | 0.4916 | 0.2265 | 0.3046 | 🟡 Hafif Overfitting |
| **XGBoost** | **3. Temel Veri (Hiperparametre Ayarlı)** | **0.7343** | **0.5493** | **0.1849** | **0.2689** | **🟡 Hafif Overfitting** |
| Artificial_Neural_Network | 3. Temel Veri (Hiperparametre Ayarlı) | 0.6257 | 0.4361 | 0.1896 | 0.3661 | 🟡 Hafif Overfitting |
| AdaBoost | 3. Temel Veri (Hiperparametre Ayarlı) | 0.7532 | 0.5024 | 0.2508 | 0.2947 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 3. Temel Veri (Hiperparametre Ayarlı) | 0.6259 | 0.3324 | 0.2935 | 0.3929 | ⚠️ Yüksek Overfitting |
| Random_Forest | 4. Gelişmiş Veri (Hiperparametre Ayarlı) | 0.7044 | 0.4964 | 0.2080 | 0.3061 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 4. Gelişmiş Veri (Hiperparametre Ayarlı) | 0.7118 | 0.4893 | 0.2225 | 0.3078 | 🟡 Hafif Overfitting |
| **XGBoost** | **4. Gelişmiş Veri (Hiperparametre Ayarlı)** | **0.7502** | **0.5498** | **0.2004** | **0.2693** | **🟡 Hafif Overfitting** |
| Artificial_Neural_Network | 4. Gelişmiş Veri (Hiperparametre Ayarlı) | 0.6449 | 0.4082 | 0.2368 | 0.3891 | 🟡 Hafif Overfitting |
| AdaBoost | 4. Gelişmiş Veri (Hiperparametre Ayarlı) | 0.7440 | 0.5147 | 0.2292 | 0.2856 | 🟡 Hafif Overfitting |
| Decision_Tree | 4. Gelişmiş Veri (Hiperparametre Ayarlı) | 0.6439 | 0.3236 | 0.3204 | 0.3993 | ⚠️ Yüksek Overfitting |