import os
import re
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from sqlalchemy import create_engine, text

from langchain_singlestore.vectorstores import SingleStoreVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables from .env
load_dotenv()

# Extract configs
openai_key = os.getenv("OPENAI_API_KEY")
singlestore_url = os.getenv("SINGLESTOREDB_URL")      # for LangChain
sqlalchemy_url = os.getenv("SQLALCHEMY_DB_URL")        # for SQLAlchemy inserts

# Validate
if not openai_key:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")
if not singlestore_url:
    raise ValueError("❌ Missing SINGLESTOREDB_URL in .env")
if not sqlalchemy_url:
    raise ValueError("❌ Missing SQLALCHEMY_DB_URL in .env")

# Set the env var LangChain expects
os.environ["SINGLESTOREDB_URL"] = singlestore_url


# --- STEP 1: Parse document ---
def extract_text_from_document(file_path):
    print(f"📄 Reading document: {file_path}")
    elements = partition(filename=file_path)
    chunks = [el.text for el in elements if el.text and len(el.text.split()) > 50]
    print(f"✅ Extracted {len(chunks)} valid chunks")
    return chunks


# --- STEP 2: Clean each chunk ---
def clean_chunk(text):
    if not text:
        return ""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', '', text)
    return text.strip()


# --- STEP 3: Store raw text in SQL table ---
def setup_sql_table_and_insert(chunks):
    engine = create_engine(sqlalchemy_url)
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            content LONGTEXT
        )
        """))

        inserted = 0
        for chunk in chunks:
            safe_chunk = clean_chunk(chunk)
            if safe_chunk:
                try:
                    conn.execute(
                        text("INSERT INTO documents (content) VALUES (:text)"),
                        {"text": safe_chunk}
                    )
                    inserted += 1
                except Exception as e:
                    print(f"⚠️ Skipped chunk due to DB error: {str(e)}")

        print(f"✅ Inserted {inserted} clean chunks into SQL table")


# --- STEP 4: Store embeddings in LangChain vector store ---
def store_vectors(chunks):
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = SingleStoreVectorStore(embedding=embedding_model)
    docs = [Document(page_content=clean_chunk(c)) for c in chunks]
    vectorstore.add_documents(docs)
    print("✅ Vector embeddings stored in SingleStore")
    return vectorstore


# --- STEP 5: Query agent loop ---
def query_agent(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    print("\n🤖 Ask questions about the document. Type 'exit' to quit.")
    while True:
        q = input("💬 Your Question: ")
        if q.strip().lower() in ("exit", "quit"):
            print("👋 Exiting.")
            break
        answer = qa_chain.run(q)
        print("🧠 Answer:", answer)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    file_path = "EnviStats_India_2024.pdf"
    chunks = extract_text_from_document(file_path)
    setup_sql_table_and_insert(chunks)
    vectorstore = store_vectors(chunks)
    query_agent(vectorstore)