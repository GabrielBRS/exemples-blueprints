def retrieve(state: GraphState):
    """Recupera documentos do Vector Store baseados na pergunta."""
    print("--- RECUPERANDO DOCUMENTOS ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState):
    """Filtra os documentos recuperados, removendo os irrelevantes."""
    print("--- AVALIANDO RELEVÂNCIA DOS DOCUMENTOS ---")
    question = state["question"]
    documents = state["documents"]

    # Prompt para o LLM atuar como um classificador binário
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um avaliador que verifica se um documento é relevante para a pergunta do usuário. Responda apenas com 'sim' ou 'nao'."),
        ("human", "Pergunta: {question}\n\nDocumento: {document}")
    ])
    grader_chain = prompt | llm | StrOutputParser()

    filtered_docs = []
    for d in documents:
        score = grader_chain.invoke({"question": question, "document": d.page_content})
        if "sim" in score.lower():
            filtered_docs.append(d)
        else:
            print(f"Documento irrelevante descartado.")

    return {"documents": filtered_docs, "question": question}


def generate(state: GraphState):
    """Gera a resposta final usando os documentos filtrados."""
    print("--- GERANDO RESPOSTA ---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Use o seguinte contexto para responder à pergunta. Se não souber, diga que não sabe.\n\nContexto: {context}"),
        ("human", "{question}")
    ])
    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def transform_query(state: GraphState):
    """Reescreve a pergunta se os documentos recuperados não forem bons o suficiente."""
    print("--- TRANSFORMANDO A PERGUNTA ---")
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um otimizador de buscas. Analise a pergunta do usuário e reescreva-a para melhorar a recuperação em um banco de dados vetorial."),
        ("human", "Pergunta original: {question}")
    ])
    rewrite_chain = prompt | llm | StrOutputParser()

    better_question = rewrite_chain.invoke({"question": question})
    return {"documents": state["documents"], "question": better_question}