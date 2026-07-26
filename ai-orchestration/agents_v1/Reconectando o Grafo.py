from langgraph.graph import StateGraph, END

workflow = StateGraph(GraphState)

# Adiciona todos os nós
workflow.add_node("retrieve", retrieve) # Seu nó de busca original
workflow.add_node("grade_documents", grade_documents) # Seu avaliador original
workflow.add_node("generate", generate) # Seu gerador original
workflow.add_node("transform_query", transform_query)
workflow.add_node("max_retries_fallback", max_retries_fallback) # NOVO NÓ

# Define o fluxo de arestas condicionais atualizado
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "fallback": "max_retries_fallback" # NOVA ROTA
    }
)

# Arestas estáticas
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("max_retries_fallback", END) # Fallback encerra o grafo

app = workflow.compile()

# IMPORTANTE: Inicialize o contador com 0 na chamada inicial!
inputs = {"question": "O que é o LangGraph?", "loop_count": 0}

for output in app.stream(inputs):
    # O output fluirá respeitando o limite rígido de 5 tentativas
    pass