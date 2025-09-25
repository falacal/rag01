#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import glob
import chardet
import shutil
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableSequence
from langchain.schema import Document

# ----------------------------
# AYARLAR
# ----------------------------
DOCUMENTS_DIR = "C:/FIZIK_RAG/RAG_documents"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 10
VECTOR_DB_PATH = "vector_db"

# Model ve Embedding Ayarları
llm = OllamaLLM(
    model="qwen3:8b",
    temperature=0.7,
    num_predict=3000,
    system="Kısa ve öz cevaplar ver. Sadece belgelerdeki bilgiye göre yanıtla."
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": False}
)

# ----------------------------
# TÜRKÇE KARAKTER DÜZELTME
# ----------------------------
def fix_turkish_text(text: str) -> str:
    replacements = {
        '─▒': 'ı', '├╝': 'ü', '├ö': 'ö', '├ğ': 'ğ',
        '├ş': 'ş', '├ç': 'ç', '─×': 'Ğ', '─░': 'İ',
        '├ı': 'ı', '├º': 'ş', '├®': 'é', '├ñ': 'ä',
        '├á': 'á', '├¡': 'í',
    }
    for broken, correct in replacements.items():
        text = text.replace(broken, correct)
    return text

def detect_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8'

# ----------------------------
# BELGE YÜKLEME
# ----------------------------
def load_text_file_with_encoding(filepath: str) -> List[Document]:
    try:
        encoding = detect_encoding(filepath)
        loader = TextLoader(filepath, encoding=encoding)
        docs = loader.load()
        for doc in docs:
            doc.page_content = fix_turkish_text(doc.page_content)
        return docs
    except Exception as e:
        logging.error(f"TXT yükleme hatası ({filepath}): {e}")
        return []

def load_pdf_with_fix(filepath: str) -> List[Document]:
    try:
        loader = PyMuPDFLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.page_content = fix_turkish_text(doc.page_content)
        return docs
    except Exception as e:
        logging.error(f"PDF yükleme hatası ({filepath}): {e}")
        return []

def load_local_documents() -> List[Document]:
    documents = []

    # 1. ÖNCELİKLİ: oku.txt dosyasını ayrı yükle
    oku_txt_path = os.path.join(DOCUMENTS_DIR, "oku.txt")
    if os.path.exists(oku_txt_path):
        oku_docs = load_text_file_with_encoding(oku_txt_path)
        if oku_docs:
            # Her bir satırı ayrı bir belge olarak işle
            lines = oku_docs[0].page_content.split('\n')
            for line in lines:
                if line.strip():  # Boş satırları atla
                    documents.append(Document(page_content=line.strip()))
            logging.info(f"ÖZEL: oku.txt'den {len(documents)} tanım yüklendi")
            print(f"📄 oku.txt içeriği:")
            for i, doc in enumerate(documents):
                print(f"  {i+1}. {doc.page_content}")

    # 2. Diğer belgeleri normal şekilde yükle
    pdf_files = glob.glob(os.path.join(DOCUMENTS_DIR, "**/*.pdf"), recursive=True)
    txt_files = glob.glob(os.path.join(DOCUMENTS_DIR, "**/*.txt"), recursive=True)
    word_files = glob.glob(os.path.join(DOCUMENTS_DIR, "**/*.doc*"), recursive=True)

    # PDF'leri yükle
    for pdf_file in pdf_files:
        docs = load_pdf_with_fix(pdf_file)
        documents.extend(docs)
        logging.info(f"Yüklendi: {pdf_file} ({len(docs)} sayfa)")

    # TXT'leri yükle (oku.txt hariç)
    for txt_file in txt_files:
        if txt_file == oku_txt_path:
            continue
        docs = load_text_file_with_encoding(txt_file)
        documents.extend(docs)
        logging.info(f"Yüklendi: {txt_file} ({len(docs)} belge)")

    # Word belgelerini yükle
    for word_file in word_files:
        try:
            loader = UnstructuredWordDocumentLoader(word_file)
            docs = loader.load()
            for doc in docs:
                doc.page_content = fix_turkish_text(doc.page_content)
            documents.extend(docs)
            logging.info(f"Yüklendi: {word_file} ({len(docs)} belge)")
        except Exception as e:
            logging.error(f"Word yükleme hatası ({word_file}): {e}")

    return documents

# ----------------------------
# VEKTÖR VERİTABANI
# ----------------------------
def create_local_vector_db() -> Optional[FAISS]:
    documents = load_local_documents()
    if not documents:
        logging.error("Hiçbir belge yüklenemedi!")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    splits = splitter.split_documents(documents)
    logging.info(f"Toplam {len(documents)} belgeden {len(splits)} parça oluşturuldu")

    try:
        vector_db = FAISS.from_documents(splits, embeddings)
        logging.info("Vektör veritabanı başarıyla oluşturuldu")
        return vector_db
    except Exception as e:
        logging.error(f"Vektör veritabanı oluşturma hatası: {e}")
        return None

# ----------------------------
# DISK İŞLEMLERİ
# ----------------------------
def save_vector_db(db: FAISS, path: str):
    """FAISS veritabanını diske kaydeder"""
    if not os.path.exists(path):
        os.makedirs(path)
    db.save_local(path)
    print(f"✅ FAISS veritabanı '{path}' konumuna kaydedildi.")

def load_vector_db(path: str) -> FAISS:
    """FAISS veritabanını diskten yükler"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS veritabanı '{path}' bulunamadı.")
    print(f"📦 FAISS veritabanı '{path}' konumundan yükleniyor...")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_or_load_vector_db(force_recreate=False) -> Optional[FAISS]:
    """Veritabanını diskten yükler veya oluşturur"""
    if force_recreate and os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print(f"🗑️ Mevcut veritabanı silindi, yeniden oluşturulacak.")
    
    if os.path.exists(VECTOR_DB_PATH):
        try:
            return load_vector_db(VECTOR_DB_PATH)
        except Exception as e:
            logging.error(f"Veritabanı yükleme hatası: {e}")
    
    # Veritabanı yoksa veya yüklenemezse oluştur
    db = create_local_vector_db()
    if db:
        save_vector_db(db, VECTOR_DB_PATH)
    return db

# ----------------------------
# ÖNCELİKLENDİRME
# ----------------------------
def prioritize_results(query: str, docs: List[Document]) -> List[Document]:
    """
    Belgeleri önceliklendirir:
    1. Tam başlık eşleşmeleri (ör: "hız:")
    2. Kısmi başlık eşleşmeleri (ör: "hız")
    3. Diğer belgeler
    """
    if not docs:
        return []
    
    # Sorgudan anahtar kelimeleri çıkar
    query_lower = query.lower().rstrip("?").replace(" nedir", "").strip()
    
    # Tam başlık eşleşmesi için format (ör: "hız:")
    exact_keyword = f"{query_lower}:"
    
    # Öncelik sırasına göre belgeleri ayır
    exact_matches = []
    partial_matches = []
    others = []
    
    print(f"\n🔍 Önceliklendirme için aranan anahtar kelimeler:")
    print(f"  Tam eşleşme: '{exact_keyword}'")
    print(f"  Kısmi eşleşme: '{query_lower}'")
    
    for doc in docs:
        content_lower = doc.page_content.lower()
        
        if exact_keyword in content_lower:
            print(f"✅ Tam eşleşme bulundu: '{doc.page_content[:50]}...'")
            exact_matches.append(doc)
        elif query_lower in content_lower:
            print(f"⚠️ Kısmi eşleşme bulundu: '{doc.page_content[:50]}...'")
            partial_matches.append(doc)
        else:
            others.append(doc)
    
    print(f"\n📊 Önceliklendirme sonuçları:")
    print(f"  Tam eşleşme: {len(exact_matches)} belge")
    print(f"  Kısmi eşleşme: {len(partial_matches)} belge")
    print(f"  Diğer: {len(others)} belge")
    
    # Öncelik sırasına göre birleştir
    return exact_matches + partial_matches + others

# ----------------------------
# ARAMA
# ----------------------------
def search_local_knowledge(query: str, k: int = TOP_K) -> str:
    if not local_vector_db:
        return ""
    try:
        fixed_query = fix_turkish_text(query)
        docs = local_vector_db.similarity_search(fixed_query, k=k)
        
        print(f"\n📝 Bulunan {len(docs)} belge (önceliklendirme öncesi):")
        for i, doc in enumerate(docs):
            print(f"  {i+1}. {doc.page_content[:50]}...")
        
        prioritized = prioritize_results(fixed_query, docs)
        
        print(f"\n⭐ Önceliklendirilmiş {len(prioritized)} belge:")
        for i, doc in enumerate(prioritized):
            print(f"  {i+1}. {doc.page_content[:50]}...")
        
        return "\n---\n".join(doc.page_content for doc in prioritized)
    except Exception as e:
        logging.error(f"Arama hatası: {e}")
        return ""

# ----------------------------
# CEVAP ÜRETME
# ----------------------------
def generate_answer(query: str, local_context: str) -> str:
    if not local_context.strip():
        return "Bilmiyorum"

    prompt = PromptTemplate.from_template(
        """Aşağıdaki belgelere dayanarak Türkçe olarak soruyu yanıtla:

BELGELER:
{local_context}

SORU: {query}

Kurallar:
- Sadece belgelerdeki bilgilere göre yanıtla
- Cevap belgelerde yoksa kesinlikle "Bilmiyorum" de
- Kısa ve öz cevap ver
- Ek bilgi veya açıklama ekleme"""
    )

    chain = RunnableSequence(
        prompt | llm | StrOutputParser()
    )

    try:
        return chain.invoke({"query": query, "local_context": local_context})
    except OutputParserException as e:
        logging.error(f"Çıktı işleme hatası: {e}")
        return "Cevap oluşturulurken hata oluştu"

# ----------------------------
# ANA RAG PIPELINE
# ----------------------------
def local_rag_pipeline(query: str) -> str:
    try:
        local_context = search_local_knowledge(query)
        return generate_answer(query, local_context)
    except Exception as e:
        logging.error(f"RAG işlem hatası: {e}")
        return f"Hata: {str(e)}"

# ----------------------------
# TESTLER
# ----------------------------
def test_embedding():
    test_text = "hız nedir"
    embedding = embeddings.embed_query(test_text)
    print(f"Embedding boyutu: {len(embedding)}")
    print(f"Örnek değerler: {embedding[:5]}")
    return True

def test_turkish_text():
    broken_text = "h─▒z: Bir cismin birim zamandaki yer de─şi┼ßtirmesiyle tan─▒mlan─▒r."
    fixed_text = fix_turkish_text(broken_text)
    print(f"Bozuk: {broken_text}")
    print(f"Düzeltilmiş: {fixed_text}")
    return "hız" in fixed_text

# ----------------------------
# ANA PROGRAM
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n[Vektör Tabanlı RAG Sistemi (Türkçe Destekli)]")
    print("Model: qwen3:8b")
    print("Embeddings: sentence-transformers/all-MiniLM-L6-v2\n")

    print("=== SİSTEM TESTLERİ ===")
    print("1. Embedding testi:", "Başarılı" if test_embedding() else "Başarısız")
    print("2. Türkçe karakter testi:", "Başarılı" if test_turkish_text() else "Başarısız")
    print("=======================\n")

    # Veritabanını diskten yükle veya oluştur ve kaydet
    # force_recreate=True ile mevcut veritabanını silip yeniden oluştur
    local_vector_db = create_or_load_vector_db(force_recreate=True)

    while True:
        try:
            query = input("Soru ('çık' yazarsan çıkılır): ").strip()
            if query.lower() in ("çık", "exit", "quit"):
                break

            print("\n🔄 Cevap oluşturuluyor...")
            response = local_rag_pipeline(query)
            print(f"\n🔎 Cevap:\n{response}\n")

        except KeyboardInterrupt:
            print("\nKullanıcı tarafından durduruldu.")
            break
        except Exception as e:
            print(f"Beklenmeyen hata: {e}")
            break
