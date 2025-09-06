# Desafio MBA Engenharia de Software com IA - Full Cycle

Esta é uma solução para o desafio de Ingestão e Busca Semântica com LangChain e Postgres.

## Funcionalidades

- **Ingestão de PDF**: Lê um arquivo PDF, o divide em partes, gera embeddings e os armazena em um banco de dados Postgres com a extensão pgvector.
- **Busca Semântica**: Permite que um usuário faça perguntas via CLI e obtenha respostas baseadas unicamente no conteúdo do PDF.
- **Suporte a Múltiplos Provedores**: Suporta tanto **Google Gemini** quanto **OpenAI** para a geração de embeddings e para o modelo de linguagem, configurável através de uma variável de ambiente.

## Como executar a solução

### Pré-requisitos

- Docker e Docker Compose
- Python 3.9+
- Uma chave de API do Google (Gemini) e/ou da OpenAI.

### Passos para execução

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/danieeelfr/mba-full-cycle-ia-challenge-1.git
    cd mba-full-cycle-ia-challenge-1
    ```

2.  **Crie e ative um ambiente virtual (venv):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure suas variáveis de ambiente:**

    Crie um arquivo `.env` na raiz do projeto (você pode copiar o `.env.example`) e **preencha as variáveis necessárias**. Veja a seção abaixo para mais detalhes.
    ```bash
    cp .env.example .env
    ```

5.  **Inicie o banco de dados:**

    ```bash
    docker compose up -d
    ```

6.  **Execute a ingestão de dados:**

    Este script irá ler o PDF, processá-lo e armazenar os embeddings no banco de dados. **Você só precisa executar este passo uma vez.**

    ```bash
    python src/ingest.py
    ```

7.  **Inicie o chat:**

    Agora você pode fazer perguntas ao documento.

    ```bash
    python src/chat.py
    ```

    Para sair do chat, digite `exit`.

## Configuração (Variáveis de Ambiente)

- `EMBEDDING_PROVIDER`: Define qual provedor de IA usar.
  - `gemini` para usar Google Gemini (padrão).
  - `openai` para usar OpenAI.

- `PDF_PATH`: Caminho para o documento PDF que será ingerido. O padrão é `document.pdf`.

#### Google Gemini
- `GOOGLE_API_KEY`: Sua chave de API do Google AI Studio.
- `GOOGLE_EMBEDDING_MODEL`: Modelo de embedding a ser usado. Padrão: `models/embedding-001`.
- `GOOGLE_MODEL`: Modelo de chat a ser usado. Padrão: `gemini-1.5-flash-latest`.

#### OpenAI
- `OPENAI_API_KEY`: Sua chave de API da OpenAI.
- `OPENAI_EMBEDDING_MODEL`: Modelo de embedding a ser usado. Padrão: `text-embedding-3-small`.
- `OPENAI_MODEL`: Modelo de chat a ser usado. Padrão: `gpt-3.5-turbo`.

#### Banco de Dados
- `DATABASE_URL`: A connection string para o banco de dados PostgreSQL. O valor padrão, ideal para o ambiente Docker, está no arquivo `.env.example`.
- `PG_VECTOR_COLLECTION_NAME`: Nome da "coleção" para armazenar os vetores no banco de dados. Padrão: `documents`.