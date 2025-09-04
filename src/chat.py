from search import search_prompt

def main():
    """
    Starts an interactive chat session for question answering.

    This function initializes the retrieval-based question answering chain and
    enters a loop to accept user questions from the command line. The user can
    type 'exit' to end the session. Each question is sent to the RAG chain,
    and the generated answer is printed to the console.
    """
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Digite 'exit' para sair.")
    while True:
        question = input("Faça sua pergunta: ")
        if question.lower() == 'exit':
            break
        if not question.strip():
            continue

        response = chain.invoke(question)
        print(f"RESPOSTA: {response}")


if __name__ == "__main__":
    main()
