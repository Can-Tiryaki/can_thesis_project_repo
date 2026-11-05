# 🧬 Özellik Mühendisliği (Feature Engineering) Raporu

Bu klasör, temizlenmiş diyabet veri setinden tıp literatürüne uygun olarak türetilen yeni klinik özellikleri (features) ve bu özelliklerin oluşturulma gerekçelerini içermektedir.

Türetilen yeni özellikler, makine öğrenmesi modelinin HbA1c (3 aylık şeker ortalaması) tahmini yaparken sadece anlık kan şekerine odaklanmasını engeller; hastanın genel metabolik, lipid ve pankreatik kapasitesini anlamasına yardımcı olur.

---

## 🧪 1. Türetilen Klinik Özellikler ve Tıbbi Gerekçeleri

### A. Trigliserid / HDL-Kolesterol Oranı (`TG_HDL_Orani`)
*   **Formül:** $\text{Trigliserid} \div \text{HDL Kolesterol}$
*   **Tıbbi Anlamı:** Klinik tıp literatüründe bu oran, **İnsülin Direncinin (Insulin Resistance)** en pratik ve güvenilir dolaylı belirteçlerinden (surrogate marker) biri olarak kabul edilir. İnsülin direnci arttığında karaciğerde trigliserid sentezi artarken, HDL (iyi kolesterol) üretimi ve yarılanma ömrü azalır. Bu oran doğrudan HOMA-IR ve HbA1c seviyeleriyle korelasyon gösterir.
*   **Referans Gösterilebilir Yayınlar:**
    1.  *Marotta, T., et al. (2010). "Triglycerides/HDL cholesterol ratio: A simple marker of insulin resistance."*
    2.  *McLaughlin, T., et al. (2003). "Use of metabolic markers to identify overweight individuals who are insulin resistant." (Annals of Internal Medicine)*

### B. LDL / HDL-Kolesterol Oranı (`LDL_HDL_Orani`)
*   **Formül:** $\text{LDL Kolesterol} \div \text{HDL Kolesterol}$
*   **Tıbbi Anlamı:** Tıpta **"Kardiyovasküler Risk İndeksi"** veya **"Aterojenik İndeks"** olarak adlandırılır. Diyabet hastalarında yağ metabolizması da bozulur (Diyabetik Dislipidemi). Bu oran, glisemik kontrolü (HbA1c) bozuk olan bireylerin damar sertleşmesi ve kalp hastalığı eğilimini yansıtarak modele metabolik kararlılık hakkında bilgi sağlar.
*   **Referans Gösterilebilir Yayınlar:**
    1.  *Millán, J., et al. (2009). "Lipoprotein ratios: Physiological significance and clinical usefulness in cardiovascular prevention." (Clinical Investig. Arterioscl.)*
    2.  *Dobiasova, M. (2004). "Atherogenic Index of Plasma [log(Triglycerides/HDL-Cholesterol)]: Relevance to Cardiovascular Risk."*

### C. Açlık / Tokluk Kan Şekeri Oranı (`AKS_TKS_Orani`)
*   **Formül:** $\text{Açlık Kan Şekeri (AKŞ)} \div \text{Tokluk Kan Şekeri (TKŞ) 2.Saat}$
*   **Tıbbi Anlamı:** Vücudun dışarıdan gelen glukoz yüküne (yemek sonrası) verdiği pankreatik tepkiyi temsil eder. Sağlıklı bireylerde bu oran daha dengeliyken, diyabet ve prediyabet hastalarında (pankreas beta hücrelerinin insülin salgılama yanıtı zayıfladığında) tokluk şekeri hızla yükselir ve oran küçülür. 
*   **Referans Gösterilebilir Yayınlar:**
    1.  *Abdul-Ghani, M. A., et al. (2007). "The relationship between fasting and 2-h post-load plasma glucose levels during the OGTT." (The Journal of Clinical Endocrinology & Metabolism)*

### D. Glisemik Ortalama (`Glisemik_Ortalama`)
*   **Formül:** $(\text{AKŞ} + \text{TKŞ 1.Saat} + \text{TKŞ 2.Saat}) \div 3$
*   **Tıbbi Anlamı:** Hastanın o günkü ölçüm sürecindeki (açlık, 1. saat ve 2. saat tokluk) genel glukoz yükünün ortalamasını temsil eder. Tekil anlık şeker dalgalanmalarının gürültüsünü (noise) azaltarak modelin daha stabil bir genel şeker yükü öğrenmesini sağlar.

### E. Cinsiyet One-Hot Encoding (`Cinsiyet_Female`)
*   **Formül:** Kadın $\rightarrow 1$, Erkek $\rightarrow 0$
*   **Teknik Anlamı:** Kategorik olan cinsiyet değişkenini makine öğrenmesi modellerinin (XGBoost, Random Forest vb.) matematiksel olarak işleyebileceği ikili (binary) sayısal formata dönüştürür.

---

## 💾 Üretilen Veri Setleri
*   **CSV Formatı:** `mergeAllData_engineered.csv`
*   **Excel Formatı:** `mergeAllData_engineered.xlsx`
*   **Özellik Sayısı:** 10 (temizlenmiş) $\rightarrow$ **15 (zenginleştirilmiş)**
