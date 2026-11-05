# ==============================================================================
# Self-Querying (Azure) + Tam Başlık Chunk'ları + Ollama (Gen) RAG Test Script'i
# ==============================================================================

import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI  # Self-querying için Azure
from langchain_ollama import ChatOllama        # Cevap üretimi için Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# 1. PARAMETRE VE YOL TANIMLAMALARI
load_dotenv()

TEST_SET_PATH = os.getenv("TEST_SET_PATH", "diabetes_rag_test_set.csv")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db_contextual")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OUTPUT_RESULTS_PATH = os.getenv("OUTPUT_RESULTS_PATH", "outputs/rag_results_full_section.csv")

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

# Tıbbi Asistan Sistem Promptu
SYSTEM_PROMPT = """You are a professional medical AI assistant specializing in the field of diabetes.
Answer the questions accurately, concisely, and with medical precision based on the provided clinical context (guideline information).
Do not add unnecessary, out-of-context, or speculative information not present in the guidelines. Stick strictly to the facts mentioned in the context.

Clinical Context:
{context}"""

def main():
    print("="*60)
    print(" AZURE (Self-Query) + TAM BAŞLIK CHUNK'LARI + OLLAMA RAG")
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
    print("[✓] Vektör veritabanı bağlantısı başarılı.")
    
    # 4. Self-Querying için AZURE OpenAI Modeli Kurulumu
    print(f"[*] Self-Querying için Azure OpenAI kuruluyor (Deployment: {AZURE_DEPLOYMENT})...")
    azure_llm_for_query = AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0
    )
    
    print("[*] Self-Querying Retriever için metadata şeması tanımlanıyor...")
    metadata_field_info = [
        AttributeInfo(
            name="guideline",
            description="The source guideline name, e.g. 'ADA' or 'NIDDK'",
            type="string",
        ),
        AttributeInfo(
            name="header_1",
            description="The main section or chapter heading of the medical document",
            type="string",
        ),
        AttributeInfo(
            name="header_2",
            description="The sub-heading under the main chapter",
            type="string",
        ),
    ]
    document_content_description = "Brief medical guideline chunks about diabetes diagnosis and treatment."

    # Self-querying retriever (Sadece filtreyi / hedef başlığı yakalamak için kullanıyoruz)
    vector_retriever = SelfQueryRetriever.from_llm(
        llm=azure_llm_for_query,
        vectorstore=db,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        search_kwargs={"k": 5},
        verbose=True
    )
    print("[✓] Azure destekli Self-Querying Retriever başarıyla kuruldu.")
    
    # 5. Metin Üretimi (Generation) için OLLAMA Modeli Kurulumu
    print(f"[*] Metin üretimi için Ollama istemcisi kuruluyor (Model: {OLLAMA_MODEL})...")
    ollama_llm_for_generation = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    rag_chain = prompt | ollama_llm_for_generation | StrOutputParser()
    
    # 6. Soruların İşlenmesi (Self-Query ile Bulunan Başlığa Ait TÜM Chunk'ları Çekme)
    results = []
    q_col = "user_input" if "user_input" in df_test.columns else ("question" if "question" in df_test.columns else "")
    gt_col = "reference" if "reference" in df_test.columns else ("ground_truth" if "ground_truth" in df_test.columns else "")
    
    if not q_col or not gt_col:
        print("🔴 HATA: CSV dosyasında 'question/user_input' veya 'ground_truth/reference' kolonları bulunamadı.")
        return
        
    print(f"\n[*] {len(df_test)} soru işleniyor (Self-Query Filtresi -> O Başlığın Tüm Chunk'ları -> Ollama)...")
    
    for idx, row in df_test.iterrows():
        question = row[q_col]
        ground_truth = row[gt_col]
        print(f"    [{idx+1}/{len(df_test)}] Soru işleniyor: \"{question[:60]}...\"")
        
        try:
            # A. Self-Querying retriever'ı çalıştırarak filtrelenmiş belgeleri alıyoruz
            # Bu adım Azure yardımıyla doğru başlık filtresini uygular ve ilgili chunk'ları getirir
            retrieved_docs = vector_retriever.invoke(question)
            
            # B. İSTENEN KRİTİK ADIM: Bulunan başlık(lar) altındaki TÜM chunk'ları veritabanından çekmek
            # Self-Querying'in getirdiği ilk belgeden metadata'yı (guideline, header_1, header_2) alıyoruz
            if retrieved_docs:
                first_meta = retrieved_docs[0].metadata
                target_guideline = first_meta.get("guideline")
                target_h1 = first_meta.get("header_1")
                target_h2 = first_meta.get("header_2")
                
                # Chroma DB'den bu başlık kombinasyonuna uyan TÜM chunk'ları filtreleyip çekiyoruz
                where_filter = {"guideline": target_guideline}
                if target_h1 and target_h2:
                    where_filter = {"$and": [{"guideline": target_guideline}, {"header_1": target_h1}, {"header_2": target_h2}]}
                elif target_h1:
                    where_filter = {"$and": [{"guideline": target_guideline}, {"header_1": target_h1}]}
                
                # Veritabanından ilgili başlığın tüm parçalarını çekiyoruz (Limit koymadan o bölümün tamamı)
                db_results = db.get(where=where_filter)
                if db_results and "documents" in db_results and db_results["documents"]:
                    retrieved_docs = [
                        Document(page_content=doc_text, metadata=db_results["metadatas"][i])
                        for i, doc_text in enumerate(db_results["documents"])
                    ]
        except Exception as e:
            print(f"    ⚠ Self-query veya başlık filtreleme sırasında hata oluştu, standart aramaya dönülüyor: {e}")
            retrieved_docs = db.similarity_search(question, k=5)
        
        # C. Bağlam oluşturma
        contexts_list = [doc.page_content for doc in retrieved_docs]
        formatted_context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        
        # D. Ollama ile yanıt üretme
        try:
            answer = rag_chain.invoke({
                "context": formatted_context,
                "question": question
            })
        except Exception as e:
            print(f"🔴 Soru {idx+1} cevaplanırken hata oluştu (Ollama): {e}")
            answer = "ERROR: LLM generation failed."
            
        results.append({
            "question": question,
            "contexts": contexts_list,
            "answer": answer,
            "ground_truth": ground_truth
        })
        
        # Geçici kayıt
        df_temp = pd.DataFrame(results)
        df_temp["contexts"] = df_temp["contexts"].apply(json.dumps)
        os.makedirs(os.path.dirname(OUTPUT_RESULTS_PATH), exist_ok=True)
        df_temp.to_csv(OUTPUT_RESULTS_PATH, index=False, encoding="utf-8")
        
    print("\n" + "="*60)
    print("      TEST SÜRECİ BAŞARIYLA TAMAMLANDI")
    print("="*60)
    print(f"[✓] Toplam cevaplanan soru sayısı: {len(results)}")
    print(f"[✓] Kaydedilen dosya yolu: {os.path.abspath(OUTPUT_RESULTS_PATH)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()