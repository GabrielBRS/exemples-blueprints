def decide_to_generate(state: GraphState):
    print("--- DECIDINDO PRÓXIMO PASSO ---")
    filtered_documents = state["documents"]
    current_loops = state.get("loop_count", 0)  # Pega o valor, assumindo 0 se não existir

    print(f"Status do Loop: {current_loops}/5")

    # REGRA 1: O disjuntor. Se bateu 5 tentativas, força a saída de emergência.
    if current_loops >= 5:
        print("Decisão: Limite excedido. Direcionando para fallback.")
        return "fallback"

    # REGRA 2: Não tem documentos bons? Gira o loop de novo.
    elif not filtered_documents:
        print("Decisão: Nenhum documento relevante. Transformando query...")
        return "transform_query"

    # REGRA 3: Deu tudo certo, documentos são bons.
    else:
        print("Decisão: Documentos relevantes encontrados. Gerando...")
        return "generate"