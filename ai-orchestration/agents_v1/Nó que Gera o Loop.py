def transform_query(state: GraphState):
    print("--- TRANSFORMANDO A PERGUNTA E INCREMENTANDO CONTADOR ---")
    question = state["question"]

    # ... (mesmo código de reescrita usando o LLM que fizemos antes) ...

    # Simulação da query transformada
    better_question = f"{question} (versão otimizada)"

    # Retornamos os dados e passamos {"loop_count": 1}.
    # O Reducer (operator.add) vai somar isso ao valor existente!
    return {"documents": state["documents"], "question": better_question, "loop_count": 1}