# ==============================================================================
# Google Colab veya Yerel Ortamda Çalıştırılacak Basic RAG Değerlendirme Script'i
# ==============================================================================
# Bu script, basic_rag_test.py tarafından üretilen sonuçları okur,
# Ragas kütüphanesi ve Azure OpenAI (gpt-4.1-mini) kullanarak
# Faithfulness, Answer Relevance, Context Recall ve Context Precision metriklerini hesaplar.
#
# Kurulum (Gerekliyse):
# !pip install -q ragas==0.4.3 langchain==0.3.30 langchain-openai==0.3.35 datasets pandas python-dotenv

import os
import sys
import json
import time
import asyncio
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Ragas ve OpenAI
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
from openai import AsyncAzureOpenAI

# 1. PARAMETRE VE YOL TANIMLAMALARI
# Özel basic_rag_eval.env dosyasını yüklüyoruz (yoksa varsayılan .env veya basic_rag_eval.env dosyasına bakar)
env_path = (
    "basic_rag/basic_rag_eval.env"
    if os.path.exists("basic_rag/basic_rag_eval.env")
    else ("basic_rag_eval.env" if os.path.exists("basic_rag_eval.env") else ".env")
)
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

raw_input_path = os.getenv("INPUT_RESULTS_PATH", "rag_results_basic.csv")
raw_output_path = os.getenv("OUTPUT_EVAL_PATH", "rag_results_basic_evaluated.csv")

# Kök dizinden veya klasör içinden çalıştırma durumuna göre yolları akıllıca çözümlüyoruz:
if not os.path.exists(raw_input_path) and os.path.exists(os.path.join("basic_rag", raw_input_path)):
    INPUT_RESULTS_PATH = os.path.join("basic_rag", raw_input_path)
else:
    INPUT_RESULTS_PATH = raw_input_path

if not os.path.exists(os.path.dirname(raw_output_path) or ".") and os.path.exists("basic_rag"):
    OUTPUT_EVAL_PATH = os.path.join("basic_rag", raw_output_path)
else:
    OUTPUT_EVAL_PATH = raw_output_path

MAX_WORKERS = 2  # Rate limit aşımını engellemek için paralel worker sayısını düşük tutuyoruz

def sanity_check_env():
    """Çevre değişkenlerini kontrol eder, eksiklik durumunda sistemi durdurur."""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"🔴 HATA: Aşağıdaki .env değişkenleri eksik: {', '.join(missing_vars)}")
        print("Lütfen .env dosyasını kontrol edin.")
        sys.exit(1)
    print("[✓] Çevre değişkenleri yüklendi.")

def main():
    print("="*60)
    print("        BASIC RAG DEĞERLENDİRME SÜRECİ BAŞLATILIYOR")
    print("="*60)
    
    # 1. Çevre Değişkenleri ve Giriş Dosyası Kontrolü
    sanity_check_env()
    
    if not os.path.exists(INPUT_RESULTS_PATH):
        print(f"🔴 HATA: '{INPUT_RESULTS_PATH}' dosyası bulunamadı.")
        print("Lütfen önce 'basic_rag/basic_rag_test.py' scriptini çalıştırarak RAG cevaplarını üretin.")
        return
        
    # 2. Sonuç Verisinin Yüklenmesi ve Hazırlanması
    df = pd.read_csv(INPUT_RESULTS_PATH)
    print(f"[✓] Değerlendirilecek sonuç dosyası yüklendi. Satır sayısı: {len(df)}")
    
    # Ragas listeleri string olarak değil, gerçek python listesi olarak bekler.
    try:
        df["contexts"] = df["contexts"].apply(json.loads)
        print("[✓] 'contexts' kolonu başarıyla liste formatına dönüştürüldü.")
    except Exception as e:
        print(f"🔴 HATA: 'contexts' kolonu dönüştürülürken hata oluştu. (Dosya formatı bozuk olabilir): {e}")
        return
        
    # Değerlendirme metrik kolonlarını ilklendiriyoruz
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = None
            
    # Eğer önceden kısmen değerlendirilmiş bir dosya varsa kaldığımız yerden devam ediyoruz (Resume-from-checkpoint)
    if os.path.exists(OUTPUT_EVAL_PATH):
        try:
            df_eval = pd.read_csv(OUTPUT_EVAL_PATH)
            print(f"[✓] Mevcut değerlendirme dosyası bulundu, kalınan yer tespit ediliyor...")
            for col in metric_cols:
                if col in df_eval.columns:
                    df[col] = df_eval[col]
            # Kaydedilmiş contexts kolonunu tekrar python listelerine çeviriyoruz
            df["contexts"] = df_eval["contexts"].apply(json.loads)
        except Exception as e:
            print(f"[*] Mevcut değerlendirme dosyası yüklenemedi (Yeni başlanıyor): {e}")
            
    # 3. Azure OpenAI Değerlendirme Modellerinin Kurulması
    print("[*] Azure OpenAI istemcisi ve Embedding modeli kuruluyor...")
    openai_client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    wrapped_llm = llm_factory(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        client=openai_client
    )
    
    wrapped_embeddings = HuggingFaceEmbeddings(
        model="nomic-ai/nomic-embed-text-v1.5",
        device="cpu",
        trust_remote_code=True
    )
    
    run_config = RunConfig(max_workers=MAX_WORKERS, timeout=60.0)
    
    # 4. Satır Satır Değerlendirme (Artımlı Kaydetme ve Hata Koruma)
    print("\n[*] Ragas değerlendirmesi başlatılıyor (Satır satır kaydedilecektir)...")
    total_rows = len(df)
    
    for idx in range(total_rows):
        # Eğer bu satır zaten değerlendirilmişse geçiyoruz
        is_evaluated = True
        for col in metric_cols:
            if pd.isna(df.at[idx, col]) or df.at[idx, col] is None:
                is_evaluated = False
                break
                
        if is_evaluated:
            print(f"    [{idx+1}/{total_rows}] Zaten değerlendirilmiş, geçiliyor.")
            continue
            
        print(f"    [{idx+1}/{total_rows}] Soru değerlendiriliyor: \"{df.at[idx, 'question'][:50]}...\"")
        
        # Tek satırlık dataset oluşturuyoruz
        row_df = df.iloc[[idx]].copy()
        single_dataset = Dataset.from_pandas(row_df)
        
        try:
            result = evaluate(
                dataset=single_dataset,
                metrics=[
                    Faithfulness(llm=wrapped_llm),
                    AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings),
                    ContextRecall(llm=wrapped_llm),
                    ContextPrecision(llm=wrapped_llm)
                ],
                llm=wrapped_llm,
                embeddings=wrapped_embeddings,
                run_config=run_config,
                raise_exceptions=False
            )
            
            # Skorları DataFrame'e yazıyoruz
            for metric_name, score in result.items():
                df.at[idx, metric_name] = score
                
            # Skorları yazdırıyoruz
            scores_str = ", ".join([f"{k}: {v:.4f}" for k, v in result.items()])
            print(f"        [✓] Sonuçlar -> {scores_str}")
            
        except Exception as e:
            print(f"        🔴 HATA: Bu soru değerlendirilirken hata oluştu: {e}")
            
        # Her adımdan sonra artımlı olarak CSV'ye kaydediyoruz
        try:
            df_to_save = df.copy()
            df_to_save["contexts"] = df_to_save["contexts"].apply(json.dumps)
            os.makedirs(os.path.dirname(OUTPUT_EVAL_PATH), exist_ok=True)
            df_to_save.to_csv(OUTPUT_EVAL_PATH, index=False, encoding="utf-8")
        except Exception as save_err:
            print(f"        🔴 HATA: CSV kaydedilemedi: {save_err}")
            
    # 5. Sonuç Raporu ve Ortalama Skorların Yazdırılması
    print("\n" + "="*60)
    print("          DEĞERLENDİRME TAMAMLANDI")
    print("="*60)
    print(f"[✓] Rapor kaydedildi: {os.path.abspath(OUTPUT_EVAL_PATH)}")
    print("-"*60)
    print("          ORTALAMA METRİK SKORLARI:")
    print("-"*60)
    for col in metric_cols:
        avg_score = df[col].dropna().mean()
        if not pd.isna(avg_score):
            print(f"  - {col.upper():<20} : {avg_score:.4f}  ({avg_score*100:.1f}%)")
        else:
            print(f"  - {col.upper():<20} : Veri yok / Hatalı")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()