# 📊 Tekil Tabakalı Ağırlıklı Bölme Model Sonuçları (Normal Veri)
Bu tablo, temizlenmiş normal veri setinin (%80 Eğitim, %20 Test) olarak tabakalı bölünmesi ve **diyabetik hastalara 5 kat, prediyabetiklere 2 kat daha fazla hata cezası (sample weight) verilerek** GridSearchCV ile eğitilmesiyle elde edilen sonuçları göstermektedir.

| Model | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Test MSE | Test MAE | Diyabetik Test MAE | Durum Analizi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 0.6938 | 0.5306 | 0.1632 | 0.2473 | 0.3590 | 0.5178 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 0.8144 | 0.4504 | 0.3641 | 0.2896 | 0.3754 | 0.6264 | ⚠️ Yüksek Overfitting |
| XGBoost | 0.7766 | 0.4998 | 0.2768 | 0.2635 | 0.3684 | 0.5767 | ⚠️ Yüksek Overfitting |
| **Artificial_Neural_Network** | **0.6346** | **0.5431** | **0.0915** | **0.2407** | **0.3684** | **0.8182** | **🟢 Dengeli / Stabil Model** |
| AdaBoost | 0.7810 | 0.5098 | 0.2712 | 0.2583 | 0.3644 | 0.6342 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 0.6343 | 0.2814 | 0.3529 | 0.3786 | 0.4201 | 0.7027 | ⚠️ Yüksek Overfitting |