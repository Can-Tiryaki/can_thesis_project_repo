# 📊 Tekil Tabakalı Ağırlıklı Bölme Master Karşılaştırma Sonuçları
Bu tablo; train-test bölünmesi sırasında diyabetik hastalara 5 kat, prediyabetiklere 2 kat ceza ağırlığı verildiğinde elde edilen train/test sonuçlarını karşılaştırmaktadır.

| Model | Deney Senaryosu | Eğitim R² (Train) | Test R² (Test) | Fark (Gap R²) | Test MSE | Diyabetik Test MAE | Durum Analizi |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random_Forest | 3. Temel Veri (Weighted - Tabakalı) | 0.6938 | 0.5306 | 0.1632 | 0.2473 | 0.5178 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 3. Temel Veri (Weighted - Tabakalı) | 0.8144 | 0.4504 | 0.3641 | 0.2896 | 0.6264 | ⚠️ Yüksek Overfitting |
| XGBoost | 3. Temel Veri (Weighted - Tabakalı) | 0.7766 | 0.4998 | 0.2768 | 0.2635 | 0.5767 | ⚠️ Yüksek Overfitting |
| **Artificial_Neural_Network** | **3. Temel Veri (Weighted - Tabakalı)** | **0.6346** | **0.5431** | **0.0915** | **0.2407** | **0.8182** | **🟢 Dengeli / Stabil Model** |
| AdaBoost | 3. Temel Veri (Weighted - Tabakalı) | 0.7810 | 0.5098 | 0.2712 | 0.2583 | 0.6342 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 3. Temel Veri (Weighted - Tabakalı) | 0.6343 | 0.2814 | 0.3529 | 0.3786 | 0.7027 | ⚠️ Yüksek Overfitting |
| Random_Forest | 4. Gelişmiş Veri (Weighted - Tabakalı) | 0.7037 | 0.5413 | 0.1623 | 0.2416 | 0.4913 | 🟡 Hafif Overfitting |
| Gradient_Boosting | 4. Gelişmiş Veri (Weighted - Tabakalı) | 0.8202 | 0.4769 | 0.3433 | 0.2756 | 0.6835 | ⚠️ Yüksek Overfitting |
| XGBoost | 4. Gelişmiş Veri (Weighted - Tabakalı) | 0.8159 | 0.5140 | 0.3019 | 0.2560 | 0.6535 | ⚠️ Yüksek Overfitting |
| **Artificial_Neural_Network** | **4. Gelişmiş Veri (Weighted - Tabakalı)** | **0.6488** | **0.5539** | **0.0949** | **0.2350** | **0.7267** | **🟢 Dengeli / Stabil Model** |
| AdaBoost | 4. Gelişmiş Veri (Weighted - Tabakalı) | 0.7853 | 0.5083 | 0.2770 | 0.2591 | 0.6442 | ⚠️ Yüksek Overfitting |
| Decision_Tree | 4. Gelişmiş Veri (Weighted - Tabakalı) | 0.6367 | 0.4009 | 0.2359 | 0.3156 | 0.6332 | 🟡 Hafif Overfitting |