import shutil
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# .env dosyasındaki değişkenleri yükle
load_dotenv()

# ==============================================================================
# 1. ADIM: .env'den okunan checkpoint ile chunk'ları geri yükle
# ==============================================================================
checkpoint_file = os.getenv("CHECKPOINT_PATH", "all_chunks_contextual_checkpoint.pkl")

if not os.path.exists(checkpoint_file):
    print(f"🔴 HATA: '{checkpoint_file}' bulunamadı! Lütfen .env dosyasındaki 'CHECKPOINT_PATH' değerini kontrol edin.")
else:
    print(f"[*] Checkpoint dosyası okunuyor: {checkpoint_file}")
    with open(checkpoint_file, "rb") as f:
        all_chunks = pickle.load(f)
    print(f"[✓] Toplam {len(all_chunks)} chunk checkpoint'ten başarıyla yüklendi!")

    # ==========================================================================
    # 2. ADIM: Nomic Embed (v1.5) modelini GPU üzerinde yükle
    # ==========================================================================
    print("[*] Nomic Embed modeli GPU üzerinden HuggingFace'e bağlanıyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={
            "trust_remote_code": True, 
            "device": "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        }
    )
    print("[✓] Embedding modeli yüklendi ve cihaza (GPU/CPU) konumlandırıldı.")

    # ==========================================================================
    # 3. ADIM: Chroma Vektör Veri Tabanını oluştur (GPU hızlandırmalı)
    # ==========================================================================
    db_dir = "chroma_db_contextual"

    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    print("[*] Contextual Chroma veri tabanı oluşturuluyor ve embeddingler hesaplanıyor...")
    db = Chroma.from_documents(all_chunks, embeddings, persist_directory=db_dir)
    print(f"[✓] Chroma veri tabanı başarıyla oluşturuldu ve '{db_dir}' klasörüne kaydedildi.")

    # ==========================================================================
    # 4. ADIM: TEST SORGUSU
    # ==========================================================================
    query = "What is the diagnosis criteria for diabetes?"
    nomic_query = f"search_query: {query}"

    print(f"\n🔎 TEST SORGUSU: '{query}'")
    results = db.similarity_search(nomic_query, k=3)

    for idx, doc in enumerate(results, 1):
        content_clean = doc.page_content.replace("search_document: ", "", 1)
        print(f"\n[{idx}] Skor/Eşleşme:")
        print(f"    - Kaynak: {doc.metadata.get('source')}")
        print(f"    - Kılavuz: {doc.metadata.get('guideline')}")
        print(f"    - Başlıklar: {doc.metadata.get('header_1')} > {doc.metadata.get('header_2', '')}")
        print(f"    - İçerik: {content_clean[:250]}...")