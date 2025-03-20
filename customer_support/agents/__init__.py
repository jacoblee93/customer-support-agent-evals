from typing import Optional

from datetime import datetime

from langgraph.graph import StateGraph, START

from customer_support.agents.state import State
from customer_support.agents.flight import initialize_flight_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig


def initialize_main_agent(checkpointer, test_date: Optional[datetime] = None):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    flight_agent = initialize_flight_agent(llm, test_date)

    def set_user_info(state: State, config: RunnableConfig):
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        return {**state, "user_info": passenger_id}

    builder = StateGraph(State)

    builder.add_node("set_user_info", set_user_info)
    builder.add_node("flight_agent", flight_agent)
    builder.add_edge(START, "set_user_info")
    builder.add_edge("set_user_info", "flight_agent")
    return builder.compile(checkpointer=checkpointer)
