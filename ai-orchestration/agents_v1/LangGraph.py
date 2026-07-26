from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. DEFINIÇÃO DO ESTADO
# Este dicionário manterá o estado da nossa aplicação a cada passo do grafo.
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# ==========================================
# SETUP DE INFRAESTRUTURA (Mocks/Exemplos)
# ==========================================
# Inicializando o LLM (Recomendado usar modelos robustos como GPT-4o ou Claude 3.5 Sonnet para avaliação)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Simulando um Vector Store
vectorstore = Chroma.from_texts(
    texts=["O LangGraph é uma biblioteca para criar agentes com LLMs usando grafos.",
           "Python foi criado por Guido van Rossum."],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()