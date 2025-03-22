from typing import Optional

from datetime import datetime

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START
from langgraph_swarm import create_handoff_tool, create_swarm

from customer_support.agents.state import State
from customer_support.agents.subagents.flight import initialize_flight_agent
from customer_support.agents.subagents.hotel import initialize_hotel_agent
from customer_support.agents.tools import DEMO_PASSENGER_ID

# from customer_support.agents.subagents.optimized.flight import initialize_flight_agent
# from customer_support.agents.subagents.optimized.hotel import initialize_hotel_agent

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def initialize_swarm_agent_with_defaults():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return initialize_swarm_agent(llm)


def initialize_swarm_agent(
    llm: BaseChatModel,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    test_date: Optional[datetime] = None,
):
    flight_agent = initialize_flight_agent(
        llm,
        [
            create_handoff_tool(
                agent_name="hotel_agent",
                description="Transfer to Bob, a customer support agent specializing in hotels",
            )
        ],
        "flight_agent",
        test_date,
    )
    hotel_agent = initialize_hotel_agent(
        llm,
        [
            create_handoff_tool(
                agent_name="flight_agent",
                description="Transfer to Alice, a customer support agent specializing in flights",
            )
        ],
        "hotel_agent",
        test_date,
    )

    swarm = create_swarm(
        [flight_agent, hotel_agent],
        default_active_agent="flight_agent",
        state_schema=State,
    ).compile()

    def set_user_info(state: State, config: RunnableConfig):
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", DEMO_PASSENGER_ID)
        return {**state, "user_info": passenger_id}

    builder = StateGraph(State)

    builder.add_node("set_user_info", set_user_info)
    builder.add_node("swarm", swarm)
    builder.add_edge(START, "set_user_info")
    builder.add_edge("set_user_info", "swarm")
    return builder.compile(checkpointer=checkpointer)
