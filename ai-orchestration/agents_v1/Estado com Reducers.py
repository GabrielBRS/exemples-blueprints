import operator
from typing import Annotated, List, TypedDict

# 1. ESTADO ATUALIZADO
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    # O Annotated com operator.add diz ao LangGraph:
    # "Se um nó retornar {'loop_count': 1}, some 1 ao valor atual em vez de sobrescrever."
    loop_count: Annotated[int, operator.add]