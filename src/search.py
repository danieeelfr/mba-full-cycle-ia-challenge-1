import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
DB_CONNECTION = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/rag")
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

def get_llm():
    print(f"Using {EMBEDDING_PROVIDER} as LLM provider.")
    if EMBEDDING_PROVIDER == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), temperature=0)
    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash-latest"), temperature=0)

def search_prompt():
    """
    Initializes a retrieval-based question answering chain.

    This function sets up a vector store connection to a PostgreSQL database,
    configures a retriever to fetch relevant documents, and creates a RAG
    (Retrieval-Augmented Generation) chain. The chain uses a prompt template
    to combine the user's question with the retrieved context and then sends
    it to a Google Generative AI model to generate an answer.

    Returns:
        A runnable chain that can be invoked with a user's question.
    """
    embeddings = get_embeddings()
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"contexto": retriever | format_docs, "pergunta": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    return rag_chain



if __name__ == "__main__":
    chain = search_prompt()
    response = chain.invoke("Qual o faturamento da Empresa SuperTechIABrazil?")
    print(response)

