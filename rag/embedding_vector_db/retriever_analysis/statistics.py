import pandas as pd
from scipy.stats import wilcoxon

# 1. Kaydettiğimiz ana veri setini yüklüyoruz
df_res = pd.read_csv("dev_kapsamli_rag_analiz_9lu_tam.csv")

# 2. Tüm Sistemlerin Ortalama Recall ve Precision Değerlerini Otomatik Hesaplama
sistemler = {
    "1. Temel Vektör (Dense)": ('recall_basic', 'precision_basic'),
    "2. Normal Hibrit (BM25+Vec)": ('recall_hybrid', 'precision_hybrid'),
    "3. Contextual RAG (Dense)": ('recall_ctx_dense', 'precision_ctx_dense'),
    "4. Contextual + Hibrit (Saf)": ('recall_ctx_hybrid_pure', 'precision_ctx_hybrid_pure'),
    "5. Contextual + Hibrit (MQ)": ('recall_ctx_hybrid_mq', 'precision_ctx_hybrid_mq'),
    "6. Hibrit + Reranker": ('recall_rerank', 'precision_rerank'),
    "7. Ctx + Hibrit + Reranker (Saf)": ('recall_ctx_rerank_pure', 'precision_ctx_rerank_pure'),
    "8. Ctx + Hibrit + Reranker + MQ": ('recall_ctx_rerank_mq', 'precision_ctx_rerank_mq')
}

karne_listesi = []
for isim, (r_col, p_col) in sistemler.items():
    r_mean = df_res[r_col].mean() * 100
    p_mean = df_res[p_col].mean() * 100
    karne_listesi.append({"Sistem": isim, "Recall@5 (%)": round(r_mean, 1), "Precision@5 (%)": round(p_mean, 1)})

df_karne = pd.DataFrame(karne_listesi)

# 3. Wilcoxon Anlamlılık Testlerini Otomatik Hesaplama (Temel Vektör Referanslı)
wilcoxon_listesi = [
    {"Karsilastirma": "Temel vs Normal Hibrit", "p_degeri": wilcoxon(df_res['recall_basic'], df_res['recall_hybrid']).pvalue},
    {"Karsilastirma": "Temel vs Contextual Dense", "p_degeri": wilcoxon(df_res['recall_basic'], df_res['recall_ctx_dense']).pvalue},
    {"Karsilastirma": "Temel vs Contextual + Hibrit (Saf)", "p_degeri": wilcoxon(df_res['recall_basic'], df_res['recall_ctx_hybrid_pure']).pvalue},
    {"Karsilastirma": "Temel vs Contextual + Hibrit (MQ)", "p_degeri": wilcoxon(df_res['recall_basic'], df_res['recall_ctx_hybrid_mq']).pvalue},
    {"Karsilastirma": "Temel vs Hibrit + Reranker", "p_degeri": wilcoxon(df_res['recall_basic'], df_res['recall_rerank']).pvalue},
    {"Karsilastirma": "Temel vs Ctx + Hibrit + Reranker (Saf)", "p_degeri": wilcoxon(df_res['recall_basic'], df_res['recall_ctx_rerank_pure']).pvalue},
    {"Karsilastirma": "Temel vs Ctx + Hibrit + Reranker + MQ", "p_degeri": wilcoxon(df_res['recall_basic'], df_res['recall_ctx_rerank_mq']).pvalue},
]
df_wilcoxon = pd.DataFrame(wilcoxon_listesi)

# 4. Soru Tipine (Synthesizer) Göre Kırılım Tablosu
cols_to_group = [col for col in df_res.columns if 'recall_' in col]
group_df = df_res.groupby('synthesizer_name')[cols_to_group].mean() * 100
group_df['n_soru'] = df_res.groupby('synthesizer_name').size()
group_df = group_df.round(2)

# --- EKRANA YAZDIRMA ---
print("="*80)
print("🎯 OTOMATİK HESAPLANAN GENEL SİSTEM KARNESİ")
print("="*80)
print(df_karne.to_string(index=False))
print("\n" + "="*80)
print("📊 OTOMATİK HESAPLANAN WILCOXON TESTLERİ")
print("="*80)
for idx, row in df_wilcoxon.iterrows():
    print(f"{row['Karsilastirma']:<40} p-değeri : {row['p_degeri']:.5f}")
print("="*80)
print("\n🔍 SORU TİPİNE GÖRE RECALL@5 (%) KIRILIMI:")
print(group_df)
print("="*80)

# --- DOSYALARA KAYDETME ---
df_karne.to_csv("otomatik_sistem_karnesi.csv", index=False, encoding="utf-8")
df_wilcoxon.to_csv("otomatik_wilcoxon_testleri.csv", index=False, encoding="utf-8")
group_df.to_csv("otomatik_soru_tipi_kirilimi.csv", encoding="utf-8")

print("\n[✓] Tüm tablolar otomatik olarak hesaplandı ve CSV dosyaları olarak kaydedildi!")