# 🏆 HbA1c Klinik Risk Sınıflandırması MASTER DENEY RAPORU (ADA Kılavuzu)
Bu rapor; **Tabakalı Bölme**, **5-Fold Çapraz Doğrulama**, **Sınıf Ağırlıklandırmalı (Weighted)** ve **Hiperparametre Optimizasyonlu (Tuned)** tüm sınıflandırma deneylerinin özetini içermektedir.

### 👑 En Yüksek F1-Macro Başarısı Gösteren Deney:
- **Deney Tipi**: `KFold_5_Normal`
- **Model**: `AdaBoost`
- **Test F1 (Macro)**: `0.6784`
- **Test Accuracy**: `0.7395`
- **Test Recall (Macro)**: `0.6693`

## 📋 TÜM DENEYLERİN ŞAMPİYONLUK VE BAŞARI TABLOSU

| Deney Tipi | Model | Test Accuracy | Test F1 (Macro) | Precision (Macro) | Recall (Macro) | ROC-AUC (OvR) | Durum / Not |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Tabakali_Normal | Random_Forest | 0.7182 | 0.6190 | 0.6808 | 0.5848 | 0.8326 | ⚠️ Yüksek Overfitting |
| Tabakali_Normal | Gradient_Boosting | 0.6906 | 0.5991 | 0.6369 | 0.5745 | 0.8260 | ⚠️ Yüksek Overfitting |
| Tabakali_Normal | Extra_Trees | 0.7017 | 0.5112 | 0.5851 | 0.4887 | 0.8241 | ⚠️ Yüksek Overfitting |
| Tabakali_Normal | Hist_Gradient_Boosting | 0.6630 | 0.5399 | 0.5884 | 0.5145 | 0.8158 | ⚠️ Yüksek Overfitting |
| Tabakali_Normal | Artificial_Neural_Network | 0.6851 | 0.5731 | 0.5921 | 0.5602 | 0.7958 | ⚠️ Yüksek Overfitting |
| Tabakali_Normal | AdaBoost | 0.6961 | 0.5904 | 0.6559 | 0.5584 | 0.7558 | 🟢 Dengeli Model |
| Tabakali_Normal | Decision_Tree | 0.6409 | 0.5365 | 0.5820 | 0.5109 | 0.6283 | ⚠️ Yüksek Overfitting |
| Tabakali_Weighted | Random_Forest | 0.6906 | 0.5348 | 0.5789 | 0.5135 | 0.8184 | ⚠️ Yüksek Overfitting |
| Tabakali_Weighted | Gradient_Boosting | 0.6575 | 0.6032 | 0.6172 | 0.5958 | 0.8157 | ⚠️ Yüksek Overfitting |
| Tabakali_Weighted | Extra_Trees | 0.6851 | 0.5586 | 0.6328 | 0.5258 | 0.8225 | ⚠️ Yüksek Overfitting |
| Tabakali_Weighted | Hist_Gradient_Boosting | 0.6630 | 0.5505 | 0.5797 | 0.5335 | 0.8001 | ⚠️ Yüksek Overfitting |
| Tabakali_Weighted | Artificial_Neural_Network | 0.6851 | 0.5731 | 0.5921 | 0.5602 | 0.7958 | ⚠️ Yüksek Overfitting |
| Tabakali_Weighted | AdaBoost | 0.7017 | 0.6379 | 0.6433 | 0.6452 | 0.7593 | 🟢 Dengeli Model |
| Tabakali_Weighted | Decision_Tree | 0.6077 | 0.5292 | 0.5664 | 0.5053 | 0.6123 | ⚠️ Yüksek Overfitting |
| Tabakali_Tuned | Random_Forest | 0.7182 | 0.6458 | 0.6389 | 0.6535 | 0.8496 | ⚠️ Yüksek Overfitting |
| Tabakali_Tuned | Gradient_Boosting | 0.6906 | 0.5880 | 0.6510 | 0.5553 | 0.8474 | 🟡 Hafif Overfitting |
| Tabakali_Tuned | Extra_Trees | 0.7182 | 0.6783 | 0.6689 | 0.6955 | 0.8620 | 🟡 Hafif Overfitting |
| Tabakali_Tuned | AdaBoost | 0.7238 | 0.6149 | 0.6838 | 0.5799 | 0.8644 | 🟢 Dengeli Model |
| Tabakali_Tuned | Decision_Tree | 0.6354 | 0.5759 | 0.5638 | 0.6342 | 0.7823 | 🟢 Dengeli Model |
| KFold_5_Normal | Random_Forest | 0.7306 | 0.6451 | 0.6687 | 0.6353 | 0.8501 | CV Std: ±0.029 |
| KFold_5_Normal | Gradient_Boosting | 0.7117 | 0.6258 | 0.6618 | 0.6073 | 0.8435 | CV Std: ±0.051 |
| KFold_5_Normal | Extra_Trees | 0.7162 | 0.6323 | 0.6710 | 0.6111 | 0.8413 | CV Std: ±0.014 |
| KFold_5_Normal | Hist_Gradient_Boosting | 0.6907 | 0.5978 | 0.6189 | 0.5859 | 0.8397 | CV Std: ±0.028 |
| KFold_5_Normal | Artificial_Neural_Network | 0.7084 | 0.6267 | 0.6454 | 0.6170 | 0.8325 | CV Std: ±0.034 |
| **KFold_5_Normal** | **AdaBoost** | **0.7395** | **0.6784** | **0.7012** | **0.6693** | **0.7602** | CV Std: ±0.063 |
| KFold_5_Normal | Decision_Tree | 0.6608 | 0.5816 | 0.5848 | 0.5849 | 0.6830 | CV Std: ±0.028 |
| KFold_5_Weighted | Random_Forest | 0.7095 | 0.6133 | 0.6388 | 0.6050 | 0.8472 | CV Std: ±0.034 |
| KFold_5_Weighted | Gradient_Boosting | 0.7006 | 0.6457 | 0.6479 | 0.6556 | 0.8463 | CV Std: ±0.060 |
| KFold_5_Weighted | Extra_Trees | 0.7151 | 0.6120 | 0.6410 | 0.6014 | 0.8441 | CV Std: ±0.040 |
| KFold_5_Weighted | Hist_Gradient_Boosting | 0.7007 | 0.6128 | 0.6254 | 0.6096 | 0.8368 | CV Std: ±0.042 |
| KFold_5_Weighted | Artificial_Neural_Network | 0.7084 | 0.6267 | 0.6454 | 0.6170 | 0.8325 | CV Std: ±0.034 |
| KFold_5_Weighted | AdaBoost | 0.6851 | 0.6375 | 0.6181 | 0.6758 | 0.7496 | CV Std: ±0.057 |
| KFold_5_Weighted | Decision_Tree | 0.6385 | 0.5350 | 0.5527 | 0.5288 | 0.6436 | CV Std: ±0.047 |