from typing import Optional

from datetime import datetime

from langgraph.graph import StateGraph, START
from langgraph_swarm import create_handoff_tool, create_swarm

from customer_support.agents.state import State
from customer_support.agents.subagents.flight import initialize_flight_agent
from customer_support.agents.subagents.hotel import initialize_hotel_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig


def initialize_swarm_agent(checkpointer, test_date: Optional[datetime] = None):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    flight_agent = initialize_flight_agent(
        llm,
        [
            create_handoff_tool(
                agent_name="hotel_booker",
                description="Transfer to Bob, a hotel booking agent",
            )
        ],
        "flight_booker",
        test_date,
    )
    hotel_agent = initialize_hotel_agent(
        llm,
        [
            create_handoff_tool(
                agent_name="flight_booker",
                description="Transfer to Alice, a flight booking agent",
            )
        ],
        "hotel_booker",
        test_date,
    )

    swarm = create_swarm(
        [flight_agent, hotel_agent],
        default_active_agent="flight_booker",
        state_schema=State,
    ).compile()

    def set_user_info(state: State, config: RunnableConfig):
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        return {**state, "user_info": passenger_id}

    builder = StateGraph(State)

    builder.add_node("set_user_info", set_user_info)
    builder.add_node("swarm", swarm)
    builder.add_edge(START, "set_user_info")
    builder.add_edge("set_user_info", "swarm")
    return builder.compile(checkpointer=checkpointer)
