from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages,AnyMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from IPython.display import Image,display
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant")

def make_graph():
    def llm_node(state: State):
        return {'messages':[llm.invoke(state['messages'])]}
    graph_builder = StateGraph(State)
    graph_builder.add_node("agent",llm_node)
    graph_builder.add_edge(START,"agent")
    graph_builder.add_edge("agent",END)
    graph = graph_builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return graph

agent = make_graph()