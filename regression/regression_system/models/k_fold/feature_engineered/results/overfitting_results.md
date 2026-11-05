# 🔍 Aşırı Öğrenme (Overfitting) Analiz Raporu (Gelişmiş Veri)
Bu rapor, özellik mühendisliği yapılmış zenginleştirilmiş veri seti (`mergeAllData_engineered.csv`) üzerinde modellerin aşırı öğrenme (overfitting) durumunu, **Eğitim Seti (Train)** ve **Doğrulama Seti (Validation)** sonuçlarını karşılaştırarak analiz eder.

### Temel Analiz Bulguları:
*   **Overfitting Gap:** Eğitim R² ile Doğrulama R² arasındaki farktır. Bu farkın düşük olması modelin daha genelleyici ve kararlı olduğunu gösterir.
*   En iyi performansı gösteren model **koyu** harflerle işaretlenmiştir.

| Model | Eğitim R² (Train) | Doğrulama R² (Val) | Fark (Gap R²) | Eğitim MSE | Doğrulama MSE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.8745 | 0.5014 | 0.3731 | 0.0786 | 0.2969 | ⚠️ Yüksek Overfitting (Ezberleme) |
| Gradient_Boosting | 0.9139 | 0.5093 | 0.4045 | 0.0541 | 0.2883 | ⚠️ Yüksek Overfitting (Ezberleme) |
| XGBoost | 0.8815 | 0.5271 | 0.3544 | 0.0744 | 0.2801 | ⚠️ Yüksek Overfitting (Ezberleme) |
| Artificial_Neural_Network | 0.8091 | -0.2103 | 1.0194 | 0.1204 | 0.8529 | ⚠️ Yüksek Overfitting (Ezberleme) |
| **AdaBoost** | **0.7807** | **0.5332** | **0.2476** | **0.1377** | **0.2756** | **🟡 Hafif Overfitting** |
| Decision_Tree | 0.6895 | 0.3087 | 0.3808 | 0.1954 | 0.4088 | ⚠️ Yüksek Overfitting (Ezberleme) |