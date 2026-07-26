# Inicializa o grafo passando a estrutura do estado
workflow = StateGraph(GraphState)

# Adiciona os nós
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Define o fluxo principal
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Adiciona a lógica condicional
workflow.add_conditional_edges(
    "grade_documents",  # Nó de origem
    decide_to_generate,  # Função de decisão
    {
        "transform_query": "transform_query",  # Se retornar "transform_query", vai para o nó "transform_query"
        "generate": "generate"  # Se retornar "generate", vai para o nó "generate"
    }
)

# Fecha o ciclo: após transformar a query, volta para a recuperação
workflow.add_edge("transform_query", "retrieve")

# A geração é o último passo
workflow.add_edge("generate", END)

# Compila o aplicativo
app = workflow.compile()

# Executando a consulta
inputs = {"question": "O que é o LangGraph?"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Processando nó: {key}")

print("\nRESPOSTA FINAL:")
print(value["generation"])