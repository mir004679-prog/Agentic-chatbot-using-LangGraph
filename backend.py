from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import math
import requests
import sqlite3
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.types import interrupt, Command


load_dotenv()


# LLM - GROQ

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)


# Embeddings Model - HUGGING FACE

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# RAG DOCUMENT INGESTION

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(DB_PATH)


def get_retriever():

    DB_PATH = "faiss_db"

    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever


# RAG TOOL

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.
    Use this tool for questions about uploaded PDF documents.
    """

    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# TOOLS

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple mathematical calculations.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:

        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given stock symbol.
    """

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=GLOBAL_QUOTE&symbol={symbol}"
        f"&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
    )

    r = requests.get(url, timeout=10)

    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing stocks.
    Requires human approval before confirming.
    """

    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}? (yes/no)"
    )

    if isinstance(decision, str) and decision.lower() == "yes":

        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity
        }

    else:

        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity
        }


@tool
def get_current_weather(location: str) -> str:
    """
    Get current real-time weather for a given city or location.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Weather API key is missing."

    try:

        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10
        )

        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]

        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]

        description = weather_data["weather"][0]["description"]

        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")

        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out."

    except requests.HTTPError as error:

        status_code = (
            error.response.status_code
            if error.response
            else "unknown"
        )

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"


# MAKE TOOL LIST

tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock
]


# MAKE LLM TOOL-AWARE

llm_with_tools = llm.bind_tools(tools)


# STATE

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


# NODE 1 - CHAT NODE

def chat_node(state: ChatState):

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"

            "- Use rag_tool for questions about uploaded PDFs.\n"
            "- Use search_tool for current events and recent information.\n"
            "- Use calculator for mathematical calculations.\n"
            "- Use get_stock_price for current stock prices.\n"
            "- Use purchase_stock when the user wants to purchase stocks.\n"
            "- Use get_current_weather for current weather.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from uploaded documents. "
            "If the user asks about a PDF but no document is available, "
            "ask them to upload a PDF. "
            "After receiving a tool result, provide a clear final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# NODE 2 - TOOL NODE

tool_node = ToolNode(tools)


# CHECKPOINTER - SQLITE

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpoint = SqliteSaver(conn)


# GRAPH

graph = StateGraph(ChatState)


# ADD NODES

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


# ADD EDGES

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge("tools", "chat_node")


# COMPILE GRAPH

chatbot = graph.compile(
    checkpointer=checkpoint
)


# HELPER FUNCTION FOR STREAMLIT FRONTEND

def get_all_threads():

    all_threads = set()

    for ckpt in checkpoint.list(None):

        all_threads.add(
            ckpt.config["configurable"]["thread_id"]
        )

    return list(all_threads)