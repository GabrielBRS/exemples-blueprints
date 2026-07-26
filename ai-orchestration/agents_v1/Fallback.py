def max_retries_fallback(state: GraphState):
    """
    Nó de emergência acionado quando o Circuit Breaker desarma.
    Gera uma resposta padrão de falha para evitar quebrar a aplicação.
    """
    print("--- CIRCUIT BREAKER ACIONADO: Limite de 5 buscas atingido ---")

    fallback_message = (
        "Desculpe, realizei 5 buscas exaustivas na base de conhecimento, "
        "mas os documentos encontrados não são suficientemente relevantes "
        "para garantir uma resposta precisa à sua pergunta."
    )

    # Retornamos a geração com a mensagem de fallback
    return {"generation": fallback_message}