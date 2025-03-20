from typing import Optional

from datetime import datetime

from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from customer_support.agents.state import State

from customer_support.agents.tools import (
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
)


SYSTEM_PROMPT = """
You are a specialized assistant for handling hotel bookings.

Search for available hotels based on the user's preferences and confirm the booking details with the customer. 
When searching, be persistent. Expand your query bounds if the first search returns no results.

Be proactive and anticipate the user's needs based on prior conversation history.
For example, if the user mentions previously that they are arriving in a city on a certain day,
you should immediately start looking for available hotel options for that day in that city before responding.
The user will then be able to narrow down the results based on their preferences. We want to proactively show the user options
and have them choose one or clarify their preferences. However, you should still remember to respond in a conversational manner.

Remember that a booking isn't completed until after the relevant tool has successfully been used.
Do not waste the user's time. Do not make up invalid tools or functions.

You do not need to ask if the user is ok with transferring to another agent, just do it.

Current user:
<User>
{user_info}
</User>

Current time:
{time}
"""

HANDOFF_SYSTEM_PROMPT = """
You are a specialized assistant for handling hotel bookings.

You have just received an invisible handoff from another agent. Look back through the provided context and
conversation history to determine your next course of action. This will almost always involve using an available tool to fetch additional information.

Be proactive and anticipate the user's needs based on prior conversation history.
For example, if the user mentions previously that they are arriving in a city on a certain day,
you should immediately start looking for available hotel options for that day in that city before responding.
The user will then be able to narrow down the results based on their preferences. We want to proactively show the user options
and have them choose one or clarify their preferences. However, you should still remember to respond in a conversational manner.

Search for available hotels based on the user's preferences and confirm the booking details with the customer. 
When searching, be persistent. Expand your query bounds if the first search returns no results. 

Remember that a booking isn't completed until after the relevant tool has successfully been used.
Do not waste the user's time. Do not make up invalid tools or functions.

Current user:
<User>
{user_info}
</User>

Current time:
{time}
"""


def initialize_hotel_agent(
    llm: BaseChatModel,
    additional_tools: list,
    name: str,
    test_date: Optional[datetime] = None,
):
    hotel_agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=test_date if test_date else datetime.now())

    handoff_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", HANDOFF_SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=test_date if test_date else datetime.now())

    base_tools = [
        search_hotels,
        book_hotel,
        update_hotel,
        cancel_hotel,
    ]

    tools = [
        *additional_tools,
        *base_tools,
    ]

    main_agent = create_react_agent(
        llm,
        tools=tools,
        prompt=hotel_agent_prompt,
        state_schema=State,
    )

    def check_for_handoff(state: State):
        last_message = state["messages"][-1]
        if last_message.type == "tool" and last_message.name == f"transfer_to_{name}":
            return "handle_handoff"
        return "main_agent"

    # Perform at least one step of the agent with no handoff tool
    # and a more focused prompt to encourage tool calling
    def handle_handoff(state: State):
        llm_with_base_tools = llm.bind_tools(base_tools)
        tool_node = ToolNode(base_tools)
        formatted_messages = handoff_prompt.invoke(state)
        response = llm_with_base_tools.invoke(formatted_messages)
        if response.tool_calls:
            tool_responses = tool_node.invoke({"messages": [response]})
            return {"messages": [response, *tool_responses["messages"]]}
        else:
            return {"messages": [response]}

    def proceed_or_end(state: State):
        if state["messages"][-1].type == "tool":
            return "main_agent"
        else:
            return "__end__"

    builder = StateGraph(State)
    builder.add_node("main_agent", main_agent)
    builder.add_node("handle_handoff", handle_handoff)
    builder.add_conditional_edges(
        "__start__", check_for_handoff, ["handle_handoff", "main_agent"]
    )
    builder.add_conditional_edges(
        "handle_handoff", proceed_or_end, ["main_agent", "__end__"]
    )
    return builder.compile(name=name)
