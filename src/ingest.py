import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
DB_CONNECTION = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/rag")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")


def get_embeddings():
    print(f"Using {EMBEDDING_PROVIDER} as embedding provider.")
    if EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
    )


def ingest_pdf():
    """
    Ingests a PDF document into a PostgreSQL database with pgvector.

    This function loads a PDF document from the specified path, splits it into
    chunks of 1000 characters with an overlap of 150 characters, generates
    embeddings for each chunk using the Google Generative AI embeddings, and
    stores the chunks and their embeddings in a PostgreSQL database using the
    PGVector extension.
    """
    print(f"Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
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
