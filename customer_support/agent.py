from typing import Annotated, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from datetime import datetime


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from customer_support.tools import (
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    check_flight_for_upgrade_space,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
)

from customer_support.utils import create_tool_node_with_fallback


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


def initialize_graph(checkpointer, test_date: Optional[datetime] = None):
    # Haiku is faster and cheaper, but less accurate
    # llm = ChatAnthropic(model="claude-3-haiku-20240307")
    # llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=1)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    # You could swap LLMs, though you will likely want to update the prompts when
    # doing so!
    # from langchain_openai import ChatOpenAI

    # llm = ChatOpenAI(model="gpt-4-turbo-preview")

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

    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=test_date if test_date else datetime.now())

    part_1_tools = [
        TavilySearchResults(max_results=1),
        fetch_user_flight_information,
        search_flights,
        lookup_policy,
        update_ticket_to_new_flight,
        cancel_ticket,
        check_flight_for_upgrade_space,
        search_car_rentals,
        book_car_rental,
        update_car_rental,
        cancel_car_rental,
        search_hotels,
        book_hotel,
        update_hotel,
        cancel_hotel,
        search_trip_recommendations,
        book_excursion,
        update_excursion,
        cancel_excursion,
    ]
    part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

    from langgraph.graph import StateGraph, START
    from langgraph.prebuilt import tools_condition

    builder = StateGraph(State)

    # Define nodes: these do the work
    builder.add_node("assistant", Assistant(part_1_assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    return builder.compile(checkpointer=checkpointer)
