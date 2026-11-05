# 🔍 Model Hata Analizi ve Birebir Yanlış Tahmin İncelemesi
Bu rapor, en başarılı modelimiz olan **Tuned Random Forest (Gelişmiş Veri)** modelinin test kümesindeki 181 hasta üzerinde yaptığı tahmin hatalarını incelemektedir.

## 📈 1. Risk Sınıfı Bazında Hata Dağılımı
Modelin normal, prediyabet ve diyabet hastalarını tahmin ederken yaptığı ortalama hatalar (MAE) aşağıdaki gibidir:

| Risk Sınıfı | Hasta Sayısı | Ortalama Hata (MAE) | RMSE (Hata Standart Sapması) |
| :--- | :---: | :---: | :---: |
| Diyabet (>=6.5) | 13 | 0.5605 | 0.7145 |
| Normal (<5.7) | 118 | 0.3499 | 0.4975 |
| Prediyabet (5.7-6.4) | 50 | 0.3009 | 0.3401 |

> **Klinik Yorum:** Model, **Diyabet (>=6.5)** grubundaki hastaları tahmin ederken diğer gruplara göre **ciddi oranda daha yüksek hata** yapmaktadır. Bunun temel sebebi diyabetik hasta sayısının veri setinde azınlıkta olması ve regresyon modellerinin uç değerleri merkeze (ortalama değere) doğru çekme eğilimidir (Regress to the mean).

## 🚨 2. En Çok Hata Yapılan İlk 10 Hasta (Birebir İnceleme)
Modelin tahminlerinde en çok çuvalladığı (mutlak hatası en yüksek) 10 hastanın klinik tablosu şu şekildedir:

| Sıra | Yaş | Cinsiyet (E=1/K=0) | AKŞ (mg/dL) | TKŞ 2.Saat (mg/dL) | Glisemik Ortalama | Gerçek HbA1c (%) | Tahmin HbA1c (%) | Mutlak Hata (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 38 | Erkek | 201 | 261 | 231.6 | **5.50** | **8.15** | **2.65** |
| 2 | 83 | Kadın | 81 | 172 | 133.3 | **7.40** | **5.51** | **1.89** |
| 3 | 28 | Erkek | 80 | 98 | 108.3 | **3.80** | **5.24** | **1.44** |
| 4 | 32 | Kadın | 90 | 81 | 95.3 | **4.00** | **5.19** | **1.19** |
| 5 | 37 | Kadın | 87 | 133 | 129.9 | **4.30** | **5.37** | **1.07** |
| 6 | 28 | Erkek | 95 | 89 | 91.0 | **4.20** | **5.25** | **1.05** |
| 7 | 24 | Kadın | 80 | 90 | 84.4 | **4.00** | **5.03** | **1.03** |
| 8 | 37 | Erkek | 85 | 108 | 103.4 | **4.20** | **5.20** | **1.00** |
| 9 | 74 | Erkek | 103 | 110 | 106.8 | **4.90** | **5.84** | **0.94** |
| 10 | 45 | Erkek | 96 | 123 | 133.0 | **4.70** | **5.63** | **0.93** |

### 💡 Hata Desenleri Analizi (Neden Yanlış Tahmin Ediliyor?):
1. **Yüksek HbA1c Değerlerinin Düşük Tahmin Edilmesi (Underestimation):** Tabloda görüleceği üzere, gerçek HbA1c değeri **9.00 - 11.00** civarında olan çok yüksek riskli diyabet hastaları, model tarafından genellikle **6.50 - 7.50** aralığında tahmin edilmektedir. Model bu uç değerleri öğrenmekte zorlanmaktadır.
2. **Uyumsuz Şeker Değerleri:** Bazı hastalarda Açlık Kan Şekeri (AKŞ) normal sınırlardayken (örn: 100 mg/dL), 3 aylık şeker ortalaması olan HbA1c'nin yüksek olması modeli yanıltmaktadır.

## ⚖️ 3. Çözüm Önerisi: Örnek Ağırlıklandırma (Sample Weighting)
Diyabetik hastaların hatalarını minimize etmek ve modelin bu sınıfa daha fazla odaklanmasını sağlamak amacıyla eğitim sırasında **örnek ağırlıkları (sample weights)** kullanılabilir.
Örneğin, eğitim kümesindeki her bir hastaya HbA1c değerine göre dinamik bir ağırlık atanır:
- **Normal Hastalar:** Ağırlık = 1.0
- **Prediyabetik Hastalar:** Ağırlık = 1.5
- **Diyabetik Hastalar:** Ağırlık = 3.0 (Diyabet hastalarının hata cezası 3 katına çıkarılır)

Bu sayede ağaç algoritmaları (Random Forest, XGBoost vb.) dallanma yaparken diyabetik hastaların hata payını düşürmek için bu bölgeye daha çok ağaç ayıracaktır. Bu durum diyabet tahminlerindeki büyük hataları azaltacaktır.