# 🔍 Aşırı Öğrenme (Overfitting) Analiz Raporu (Normal Veri)
Bu rapor, modellerin eğitim verilerini ezberleyip ezberlemediğini (overfitting) doğrulamak amacıyla, **Eğitim Seti (Train)** ve **Doğrulama Seti (Validation)** sonuçlarını yan yana karşılaştırır.

### Temel Analiz Bulguları:
*   **Overfitting Gap:** Eğitim R² ile Doğrulama R² arasındaki farktır. Bu farkın çok yüksek olması (örn. > 0.15-0.20) aşırı öğrenmeye işaret eder.
*   En iyi performansı gösteren model **koyu** harflerle işaretlenmiştir.

| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.8686 | 0.4842 | 0.3844 | 0.0823 | 0.3055 | ⚠️ Yüksek Overfitting (Ezberleme) |
| Gradient_Boosting | 0.9021 | 0.4837 | 0.4184 | 0.0615 | 0.3038 | ⚠️ Yüksek Overfitting (Ezberleme) |
| **XGBoost** | **0.8686** | **0.5339** | **0.3347** | **0.0826** | **0.2778** | **⚠️ Yüksek Overfitting (Ezberleme)** |
| Artificial_Neural_Network | 0.7774 | 0.1430 | 0.6344 | 0.1397 | 0.5809 | ⚠️ Yüksek Overfitting (Ezberleme) |
| AdaBoost | 0.7720 | 0.5000 | 0.2720 | 0.1431 | 0.2999 | ⚠️ Yüksek Overfitting (Ezberleme) |
| Decision_Tree | 0.6843 | 0.3379 | 0.3464 | 0.1980 | 0.3914 | ⚠️ Yüksek Overfitting (Ezberleme) |