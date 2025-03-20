from typing import Optional

from datetime import datetime

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START
from langgraph_supervisor import create_supervisor

from customer_support.agents.state import State
from customer_support.agents.subagents.flight import initialize_flight_agent
from customer_support.agents.subagents.hotel import initialize_hotel_agent

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

SUPERVISOR_PROMPT = """
You are a team supervisor managing a flight booking agent and a hotel booking agent. 
For flight booking, use flight_agent. 
For hotel booking, use hotel_agent.

When transferring to another agent, you should also restate the original question to give
your subordinate the proper context. There is no need to ask for confirmation before transferring.
"""


def initialize_supervisor_agent(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    test_date: Optional[datetime] = None,
):
    flight_agent = initialize_flight_agent(
        llm,
        [],
        "flight_agent",
        test_date,
    )
    hotel_agent = initialize_hotel_agent(
        llm,
        [],
        "hotel_agent",
        test_date,
    )

    supervisor = create_supervisor(
        [flight_agent, hotel_agent],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        state_schema=State,
    ).compile()

    def set_user_info(state: State, config: RunnableConfig):
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        return {**state, "user_info": passenger_id}

    builder = StateGraph(State)

    builder.add_node("set_user_info", set_user_info)
    builder.add_node("supervisor", supervisor)
    builder.add_edge(START, "set_user_info")
    builder.add_edge("set_user_info", "supervisor")
    return builder.compile(checkpointer=checkpointer)
