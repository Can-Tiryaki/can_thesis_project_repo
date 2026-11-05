# ==============================================================================
# Google Colab'de çalıştırılacak CONTEXTUAL RAG Kodu (FİNAL)
# ==============================================================================
# Çalıştırmadan önce:
#   1. Sol menüden 'rag_data.zip' dosyasını yükleyin
#   2. Bir .env dosyası oluşturun (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
#      AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME içermeli)


import os
import re
import time
import zipfile
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import AzureOpenAI
from dotenv import load_dotenv
import threading
import pickle

load_dotenv()

# ==============================================================================
# 2. ADIM: rag_data.zip dosyasını aç
# ==============================================================================
zip_path = "rag_data.zip"
extract_dir = "rag_data_extracted"

if not os.path.exists(zip_path):
    print(f"🔴 HATA: Lütfen sol taraftaki dosya menüsünden '{zip_path}' dosyasını Colab'e yükleyin!")
else:
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("[✓] ZIP dosyası başarıyla açıldı!")

    # ==========================================================================
    # 3. ADIM: Nomic Embed (v1.5) modelini yükle
    # ==========================================================================
    print("[*] Nomic Embed modeli HuggingFace üzerinden yükleniyor (Lütfen bekleyin)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True, "device": "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"}
    )
    print("[✓] Embedding modeli yüklendi.")

    # ==========================================================================
    # 4. ADIM: Parçalayıcıları tanımla
    # ==========================================================================
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150

    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    all_chunks = []

    # section_full_text: (source, header_1, header_2) -> o bölümün TEMİZ, overlap'siz tam metni
    # (header_docs seviyesinde toplanıyor, yani 800 karakterlik bölünmeden ÖNCEKİ hal)
    section_full_text_raw = defaultdict(list)

    def get_section_key(source, metadata):
        """Gruplama anahtarı. header_1 ADA'da dolu (chapter başlığı), 
        NIDDK'de otomatik None -> iki dosya tipini de aynı mantıkla kapsar."""
        h1 = metadata.get("header_1")
        h2 = metadata.get("header_2")
        return (source, h1, h2)

    # ==========================================================================
    # 5. ADIM: NIDDK dosyasını parçala (sayfa bazlı)
    # ==========================================================================
    niddk_file_path = ""
    for root, dirs, files in os.walk(extract_dir):
        if "NIDDK_Guiding-Principles_clean.md" in files:
            niddk_file_path = os.path.join(root, "NIDDK_Guiding-Principles_clean.md")
            break

    if niddk_file_path and os.path.exists(niddk_file_path):
        print(f"[*] NIDDK belgesi işleniyor: {os.path.basename(niddk_file_path)}")
        with open(niddk_file_path, "r", encoding="utf-8") as f:
            niddk_content = f.read()

        parts = re.split(r'<!-- PAGE:\s*(\d+)\s*-->', niddk_content)
        pages = []
        if parts[0].strip():
            pages.append((7, parts[0]))
        for i in range(1, len(parts), 2):
            page_no = int(parts[i])
            page_text = parts[i+1]
            pages.append((page_no, page_text))

        for page_no, page_text in pages:
            if not page_text.strip():
                continue

            # 1. Aşama: SADECE başlıklara göre böl (henüz 800 karakter limiti yok)
            header_docs = markdown_splitter.split_text(page_text)

            # --- section_full_text_raw'ı BURADA, bölünmemiş haliyle topluyoruz ---
            for hdoc in header_docs:
                key = get_section_key("NIDDK_Guiding-Principles_clean.md", hdoc.metadata)
                section_full_text_raw[key].append(hdoc.page_content)

            # 2. Aşama: 800 karakterlik gerçek retrieval chunk'larına böl
            split_docs = text_splitter.split_documents(header_docs)

            for doc in split_docs:
                doc.metadata["source"] = "NIDDK_Guiding-Principles_clean.md"
                doc.metadata["guideline"] = "NIDDK"
                doc.metadata["page"] = page_no
                all_chunks.append(doc)
    else:
        print("🔴 NIDDK belgesi bulunamadı!")

    # ==========================================================================
    # 6. ADIM: ADA dosyalarını parçala (bölüm bölüm, sayfasız)
    # ==========================================================================
    ada_dir_path = ""
    for root, dirs, files in os.walk(extract_dir):
        if "ADA" in root and any(f.endswith(".md") for f in files):
            ada_dir_path = root
            break

    if ada_dir_path and os.path.exists(ada_dir_path):
        print(f"[*] ADA belgeleri işleniyor (Dizin: {ada_dir_path})...")
        ada_files = [f for f in os.listdir(ada_dir_path) if f.endswith(".md")]

        for ada_file in ada_files:
            file_path = os.path.join(ada_dir_path, ada_file)
            with open(file_path, "r", encoding="utf-8") as f:
                ada_content = f.read()

            header_docs = markdown_splitter.split_text(ada_content)

            for hdoc in header_docs:
                key = get_section_key(ada_file, hdoc.metadata)
                section_full_text_raw[key].append(hdoc.page_content)

            split_docs = text_splitter.split_documents(header_docs)

            for doc in split_docs:
                doc.metadata["source"] = ada_file
                doc.metadata["guideline"] = "ADA"
                doc.metadata["page"] = -1
                all_chunks.append(doc)
    else:
        print("🔴 ADA belgeleri klasörü bulunamadı!")

    # section_full_text_raw listelerini tek string'e çevir (overlap'siz, temiz)
    section_full_text = {key: "\n\n".join(parts) for key, parts in section_full_text_raw.items()}

    # İstatistikler
    niddk_chunk_count = sum(1 for doc in all_chunks if doc.metadata["guideline"] == "NIDDK")
    ada_chunk_count = sum(1 for doc in all_chunks if doc.metadata["guideline"] == "ADA")
    total_chunks = len(all_chunks)

    print("\n" + "="*60)
    print("           RAG BELGE PARÇALAMA & PARAMETRE İSTATİSTİKLERİ")
    print("="*60)
    print(f"[*] Deneysel Kurulum:")
    print(f"    - Chunk Size: {CHUNK_SIZE} karakter | Overlap: {CHUNK_OVERLAP} karakter")
    print(f"    - Embedding: nomic-ai/nomic-embed-text-v1.5 (768 boyut)")
    print("-"*60)
    print(f"[*] Dağılım:")
    print(f"    - NIDDK Chunks: {niddk_chunk_count} (%{niddk_chunk_count/total_chunks*100:.2f})")
    print(f"    - ADA Chunks:   {ada_chunk_count} (%{ada_chunk_count/total_chunks*100:.2f})")
    print(f"    - Toplam:       {total_chunks}")
    print(f"    - Benzersiz (source, header_1, header_2) bölümü: {len(section_full_text)}")
    avg_len = sum(len(t) for t in section_full_text.values()) / len(section_full_text)
    print(f"    - Ortalama bölüm uzunluğu: {avg_len:.0f} karakter")
    print("="*60 + "\n")

    # ==========================================================================
    # 6.5 ADIM: CONTEXTUAL RETRIEVAL — Azure OpenAI ile bağlam üretimi
    # ==========================================================================
# ==========================================================================
# 6.5 ADIM: CONTEXTUAL RETRIEVAL — Azure OpenAI ile bağlam üretimi
# ==========================================================================
    print("[*] CONTEXTUAL RAG: Azure OpenAI ile bağlam üretimi başlıyor...")
    context_start_time = time.time()  # EKLENDİ: süre ölçümü için

    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    failed_count = 0  # EKLENDİ: kaç chunk'ta bağlam üretimi başarısız oldu

    def build_context_prompt(doc, section_text):
        h1 = doc.metadata.get("header_1")
        h2 = doc.metadata.get("header_2")
        label_parts = [p for p in [h1, h2] if p]
        label = " > ".join(label_parts) if label_parts else "Unknown section"
        truncated_section = section_text[:20000]

        return f"""This chunk is part of a section titled '{label}'.

    <section>
    {truncated_section}
    </section>

    <chunk>
    {doc.page_content}
    </chunk>

    Write a short context (30-50 words) that situates this chunk within
    the overall section for the purposes of improving search retrieval
    of the chunk. Answer ONLY with the succinct context and nothing else."""
    rate_limit_lock = threading.Lock()
    rate_limited_until = [0]  # liste kullanıyoruz ki closure içinde değiştirilebilsin

    def contextualize_chunk(prompt, max_retries=3):
        for attempt in range(max_retries):
            # Eğer başka bir worker rate limit'e çarpıp bizi "bekletiyorsa", önce onu bekle
            wait_needed = rate_limited_until[0] - time.time()
            if wait_needed > 0:
                time.sleep(wait_needed)

            try:
                response = azure_client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e):
                    with rate_limit_lock:
                        # TÜM worker'lar için 30 saniyelik global bekleme süresi belirle
                        rate_limited_until[0] = time.time() + 30
                if attempt == max_retries - 1:
                    print(f"    ⚠ Bağlam üretimi başarısız (son deneme): {e}")
                    return ""
                time.sleep(30)
        return ""

    def process_one(idx_doc):
        idx, doc = idx_doc
        key = get_section_key(doc.metadata.get("source"), doc.metadata)
        section_text = section_full_text.get(key, doc.page_content)
        prompt = build_context_prompt(doc, section_text)
        context_desc = contextualize_chunk(prompt)
        contextualized = f"{context_desc}\n\n{doc.page_content}" if context_desc else doc.page_content
        return idx, contextualized, bool(context_desc)  # EKLENDİ: başarı/başarısızlık bilgisi

    print(f"[*] {len(all_chunks)} chunk için bağlam üretiliyor (paralel: 2 worker)...")
    print(f"[*] {len(all_chunks)} chunk için bağlam üretiliyor (paralel: 2 worker)...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_one, (i, doc)) for i, doc in enumerate(all_chunks)]
        completed = 0
        for future in as_completed(futures):
            idx, contextualized_text, success = future.result()
            all_chunks[idx].page_content = f"search_document: {contextualized_text}"
            if not success:
                failed_count += 1
            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - context_start_time
                rate = completed / elapsed
                remaining = (len(all_chunks) - completed) / rate if rate > 0 else 0
                print(f"    [{completed}/{len(all_chunks)}] tamamlandı... "
                    f"(geçen: {elapsed/60:.1f} dk, tahmini kalan: {remaining/60:.1f} dk, "
                    f"başarısız: {failed_count})")

                # EKLENDİ: her 100'de bir örnek bağlamlandırılmış chunk göster
                source = all_chunks[idx].metadata.get('source')
                h1 = all_chunks[idx].metadata.get('header_1')
                h2 = all_chunks[idx].metadata.get('header_2')
                preview = all_chunks[idx].page_content[:350]
                print(f"    --- Örnek (chunk #{idx}) ---")
                print(f"    Kaynak: {source} | Başlık: {h1} > {h2}")
                print(f"    İçerik: {preview}...")
                print()

    context_elapsed = time.time() - context_start_time  # EKLENDİ
    print(f"\n[✓] Tüm chunk'lar bağlamlandırıldı.")
    print(f"    - Toplam süre: {context_elapsed/60:.1f} dakika")
    print(f"    - Başarısız bağlam üretimi: {failed_count}/{len(all_chunks)} "
        f"(%{failed_count/len(all_chunks)*100:.1f})")

    # EKLENDİ: kalite kontrolü için 3 örnek bağlamlandırılmış chunk göster
    print("\n" + "-"*60)
    print("           ÖRNEK BAĞLAMLANDIRILMIŞ CHUNK'LAR (İlk 3)")
    print("-"*60)
    for i, doc in enumerate(all_chunks[:3]):
        print(f"\n[{i+1}] Kaynak: {doc.metadata.get('source')} | "
            f"Başlık: {doc.metadata.get('header_1')} > {doc.metadata.get('header_2', '')}")
        print(f"    {doc.page_content[:300]}...")
    print("-"*60 + "\n")

    print("[*] Güvenlik için bağlamlandırılmış chunk'lar diske kaydediliyor (checkpoint)...")
    with open("all_chunks_contextual_checkpoint.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print("[✓] Checkpoint kaydedildi: all_chunks_contextual_checkpoint.pkl")
    # ==========================================================================
    # 7. ADIM: Chroma Vektör Veri Tabanını oluştur — YENİ, ayrı bir klasöre
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
    # 8. ADIM: TEST SORGUSU
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