from typing import Annotated, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from datetime import datetime

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: int
    user_info: str


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from customer_support.tools import (
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    check_flight_for_upgrade_space,
)


def initialize_graph(checkpointer, test_date: Optional[datetime] = None):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    SYSTEM_PROMPT = """
You are a helpful customer support assistant for Swiss Airlines. 
Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
When searching, be persistent. Expand your query bounds if the first search returns no results. 
If a search comes up empty, expand your search before giving up.

Current user:
<User>
{user_info}
</User>

Current time:
{time}
"""

    flight_agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=test_date if test_date else datetime.now())

    tools = [
        fetch_user_flight_information,
        search_flights,
        lookup_policy,
        update_ticket_to_new_flight,
        cancel_ticket,
        check_flight_for_upgrade_space,
    ]

    flight_agent = create_react_agent(
        llm,
        tools=tools,
        prompt=flight_agent_prompt,
        state_schema=State,
    )

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
