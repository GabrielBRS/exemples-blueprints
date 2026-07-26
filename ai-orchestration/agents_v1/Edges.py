def decide_to_generate(state: GraphState):
    """
    Decide o próximo passo após a avaliação dos documentos.
    Se houver documentos relevantes, vai para a geração.
    Se não, transforma a query e tenta recuperar novamente.
    """
    print("--- DECIDINDO PRÓXIMO PASSO ---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        print("Decisão: Nenhum documento relevante. Transformando query...")
        return "transform_query"
    else:
        print("Decisão: Documentos relevantes encontrados. Gerando...")
        return "generate"