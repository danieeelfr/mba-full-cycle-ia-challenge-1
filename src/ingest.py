import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Environment variables
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
DB_CONNECTION = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/rag")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_embeddings():
    """
    Returns the embedding provider based on the environment variable.
    """
    print(f"Using {EMBEDDING_PROVIDER} as embedding provider.")
    if EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=OPENAI_API_KEY
        )
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
        google_api_key=GOOGLE_API_KEY
    )


def ingest_pdf():
    """
    Ingests a PDF document into a PostgreSQL database with pgvector.
    """
    # Construct the absolute path to the PDF file
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    absolute_pdf_path = project_root / PDF_PATH

    if not absolute_pdf_path.exists():
        print(f"Error: PDF file not found at {absolute_pdf_path}")
        return

    print(f"Loading PDF from {absolute_pdf_path}...")
    loader = PyPDFLoader(str(absolute_pdf_path))
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    print("Generating embeddings and storing in database...")
    embeddings = get_embeddings()
    PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
        pre_delete_collection=True,
    )
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_pdf()
