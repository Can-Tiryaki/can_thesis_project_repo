# 📊 HbA1c Klinik Risk Sınıflandırması Model Sonuçları (ADA Eşikleri)
Bu rapor, temizlenmiş hasta verilerinin (%80 Eğitim, %20 Test) olarak tabakalı (Stratified Split) şekilde bölünmesi ve klinik HbA1c risk sınıflarının (**0: Normal**, **1: Prediyabet**, **2: Diyabet**) sınıflandırma algoritmalarıyla eğitilmesi sonucu elde edilen metrikleri içermektedir.

## 📈 Sınıf Dağılımı
- **Sınıf 0 (Normal (<5.7))**: 587 hasta (%65.08)
- **Sınıf 1 (Prediyabet (5.7-6.4))**: 251 hasta (%27.83)
- **Sınıf 2 (Diyabet (>=6.5))**: 64 hasta (%7.10)


## 📋 Karşılaştırmalı Model Başarı Tablosu

| Model | Train Acc | Test Acc | Acc Gap | Train F1 (Macro) | Test F1 (Macro) | Precision (Macro) | Recall (Macro) | F1 (Weighted) | ROC-AUC (OvR) | Durum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random_Forest** | **1.0000** | **0.7182** | **0.2818** | **1.0000** | **0.6190** | **0.6808** | **0.5848** | **0.7057** | **0.8326** | **⚠️ Yüksek Overfitting** |
| Gradient_Boosting | 0.9459 | 0.6906 | 0.2553 | 0.9522 | 0.5991 | 0.6369 | 0.5745 | 0.6850 | 0.8260 | ⚠️ Yüksek Overfitting |
| Extra_Trees | 1.0000 | 0.7017 | 0.2983 | 1.0000 | 0.5112 | 0.5851 | 0.4887 | 0.6700 | 0.8241 | ⚠️ Yüksek Overfitting |
| Hist_Gradient_Boosting | 1.0000 | 0.6630 | 0.3370 | 1.0000 | 0.5399 | 0.5884 | 0.5145 | 0.6477 | 0.8158 | ⚠️ Yüksek Overfitting |
| Artificial_Neural_Network | 0.9334 | 0.6851 | 0.2483 | 0.9220 | 0.5731 | 0.5921 | 0.5602 | 0.6739 | 0.7958 | ⚠️ Yüksek Overfitting |
| AdaBoost | 0.7406 | 0.6961 | 0.0445 | 0.7053 | 0.5904 | 0.6559 | 0.5584 | 0.6901 | 0.7558 | 🟢 Dengeli Model |
| Decision_Tree | 1.0000 | 0.6409 | 0.3591 | 1.0000 | 0.5365 | 0.5820 | 0.5109 | 0.6337 | 0.6283 | ⚠️ Yüksek Overfitting |

## 🏆 En Başarılı Model Detaylı Sınıf Raporu: Random_Forest

```text
                      precision    recall  f1-score   support

       Normal (<5.7)       0.78      0.87      0.82       118
Prediyabet (5.7-6.4)       0.51      0.42      0.46        50
     Diyabet (>=6.5)       0.75      0.46      0.57        13

            accuracy                           0.72       181
           macro avg       0.68      0.58      0.62       181
        weighted avg       0.70      0.72      0.71       181

```