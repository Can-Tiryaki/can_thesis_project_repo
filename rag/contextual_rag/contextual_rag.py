# ==============================================================================
# Google Colab veya Yerel Ortamda Çalıştırılacak Hybrid RAG Test Script'i
# ==============================================================================
# Bu script, Ragas ile ürettiğimiz test setindeki soruları okur,
# Chroma DB'deki verileri kullanarak hem BM25 (anahtar kelime) hem de vektör araması yapar,
# iki arama sonucunu RRF (Reciprocal Rank Fusion) ile birleştirir (Ensemble),
# yerel Ollama (Gemma) modeli ile cevap üretir ve sonuçları CSV olarak kaydeder.
#
# Kurulum (Gerekliyse):
# !pip install -q langchain-ollama langchain-community pandas chromadb sentence-transformers rank-bm25

import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# 1. PARAMETRE VE YOL TANIMLAMALARI
load_dotenv() # .env dosyasındaki değişkenleri yüklüyoruz

# .env dosyasında tanımlıysa oradan okur, yoksa sağdaki varsayılan (default) değerleri kullanır:
TEST_SET_PATH = os.getenv("TEST_SET_PATH", "diabetes_rag_test_set.csv")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db_contextual")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OUTPUT_RESULTS_PATH = os.getenv("OUTPUT_RESULTS_PATH", "outputs/rag_results_hybrid.csv")

# Tıbbi Asistan Sistem Promptu
SYSTEM_PROMPT = """You are a professional medical AI assistant specializing in the field of diabetes.
Answer the questions accurately, concisely, and with medical precision based on the provided clinical context (guideline information).
Do not add unnecessary, out-of-context, or speculative information not present in the guidelines. Stick strictly to the facts mentioned in the context.

Clinical Context:
{context}"""

def main():
    print("="*60)
    print("         HYBRID RAG TEST SÜRECİ BAŞLATILIYOR")
    print("="*60)
    
    # 2. Test Setinin Yüklenmesi
    if not os.path.exists(TEST_SET_PATH):
        print(f"🔴 HATA: '{TEST_SET_PATH}' dosyası bulunamadı.")
        return
        
    df_test = pd.read_csv(TEST_SET_PATH)
    print(f"[✓] Test veri seti yüklendi. Soru sayısı: {len(df_test)}")
    
    # 3. Chroma DB ve Embedding Modelinin Yüklenmesi
    print("[*] Nomic Embed (v1.5) modeli yükleniyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True}
    )
    
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"🔴 HATA: '{CHROMA_DB_DIR}' klasörü bulunamadı. Lütfen önce RAG veritabanını oluşturun.")
        return
        
    print("[*] Chroma vektör veritabanına bağlanılıyor...")
    db = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings
    )
    vector_retriever = db.as_retriever(search_kwargs={"k": 5})
    print("[✓] Vektör veritabanı bağlantısı başarılı.")
    
    # 4. BM25 Arama Dizininin Hazırlanması
    print("[*] BM25 arama dizini için veritabanındaki tüm belgeler yükleniyor...")
    try:
        db_data = db.get()
        documents = []
        if db_data and "documents" in db_data:
            for idx, doc_text in enumerate(db_data["documents"]):
                meta = db_data["metadatas"][idx] if "metadatas" in db_data and db_data["metadatas"] else {}
                documents.append(Document(page_content=doc_text, metadata=meta))
        print(f"[✓] Toplam {len(documents)} belge başarıyla yüklendi.")
    except Exception as e:
        print(f"🔴 HATA: Veritabanı belgeleri yüklenirken hata oluştu: {e}")
        return
        
    if not documents:
        print("🔴 HATA: Veritabanında arama yapılacak belge bulunamadı.")
        return
        
    print("[*] BM25 arama dizini oluşturuluyor...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    
    # EnsembleRetriever ile Hibrit Arama Yapılandırılması (%50 BM25 + %50 Vektör)
    print("[*] Ensemble (Hibrit) Retriever kuruluyor...")
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    print("[✓] Hibrit arama motoru başarıyla kuruldu.")
    
    # 5. Ollama LLM ve Zincir (Chain) Yapılandırması
    print(f"[*] Ollama istemcisi kuruluyor (Model: {OLLAMA_MODEL})...")
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    rag_chain = prompt | llm | StrOutputParser()
    
    # 6. Soruların Hibrit RAG ile Cevaplanması
    results = []
    q_col = "user_input" if "user_input" in df_test.columns else ("question" if "question" in df_test.columns else "")
    gt_col = "reference" if "reference" in df_test.columns else ("ground_truth" if "ground_truth" in df_test.columns else "")
    
    if not q_col or not gt_col:
        print("🔴 HATA: CSV dosyasında 'question/user_input' veya 'ground_truth/reference' kolonları bulunamadı.")
        return
        
    print(f"\n[*] {len(df_test)} soru HİBRİT RAG sistemi tarafından cevaplanıyor. Lütfen bekleyin...")
    
    for idx, row in df_test.iterrows():
        question = row[q_col]
        ground_truth = row[gt_col]
        print(f"    [{idx+1}/{len(df_test)}] Soru işleniyor: \"{question[:60]}...\"")
        nomic_query = f"search_query: {question}"
        retrieved_docs = retriever.invoke(nomic_query)
        # A. Hibrit arama ile bağlamı getiriyoruz (Ensemble Retrieval)
        retrieved_docs = retriever.invoke(question)
        
        # B. Ragas için bağlam listesi oluşturuyoruz
        contexts_list = [doc.page_content for doc in retrieved_docs]
        formatted_context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        
        # C. LLM ile cevap üretiyoruz
        try:
            answer = rag_chain.invoke({
                "context": formatted_context,
                "question": question
            })
        except Exception as e:
            print(f"🔴 Soru {idx+1} cevaplanırken hata oluştu (Ollama bağlantısını kontrol edin): {e}")
            answer = "ERROR: LLM generation failed."
            
        results.append({
            "question": question,
            "contexts": contexts_list,
            "answer": answer,
            "ground_truth": ground_truth
        })
        
        # Anlık olarak geçici sonuçları kaydediyoruz (Veri kaybını önlemek için)
        df_temp = pd.DataFrame(results)
        df_temp["contexts"] = df_temp["contexts"].apply(json.dumps)
        os.makedirs(os.path.dirname(OUTPUT_RESULTS_PATH), exist_ok=True)
        df_temp.to_csv(OUTPUT_RESULTS_PATH, index=False, encoding="utf-8")
        
    print("\n" + "="*60)
    print("          HYBRID RAG CEVAP ÜRETİMİ BAŞARIYLA TAMAMLANDI")
    print("="*60)
    print(f"[✓] Toplam cevaplanan soru sayısı: {len(results)}")
    print(f"[✓] Kaydedilen dosya yolu: {os.path.abspath(OUTPUT_RESULTS_PATH)}")
    print("-"*60)
    print("[*] RAG Sisteminden ilk örnek:")
    print(f"  - Soru: {results[0]['question']}")
    print(f"  - Modelin Cevabı (Answer): {results[0]['answer'][:150]}...")
    print(f"  - Kılavuz Cevabı (Ground Truth): {results[0]['ground_truth'][:150]}...")
    print("="*60 + "\n")
    print("İpucu: Bu çıktı dosyasını doğrudan Ragas Evaluate scriptine sokarak Hibrit modelinizi değerlendirebilirsiniz.")

if __name__ == "__main__":
    main()
