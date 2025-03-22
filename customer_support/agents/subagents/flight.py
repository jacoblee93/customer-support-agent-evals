from typing import Optional

from datetime import datetime

from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_core.prompts import ChatPromptTemplate

from customer_support.agents.state import State

from customer_support.agents.tools import (
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    check_flight_for_upgrade_space,
)


def initialize_flight_agent(
    llm: BaseChatModel,
    additional_tools: list,
    name: str,
    test_date: Optional[datetime] = None,
):
    SYSTEM_PROMPT = """
You are a helpful customer support assistant for Swiss Airlines. 
Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
When searching, be persistent. Expand your query bounds if the first search returns no results. 
If a search comes up empty, expand your search before giving up.

Try to anticipate the user's needs based on the current conversation rather than asking redundant questions.

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
        *additional_tools,
        fetch_user_flight_information,
        search_flights,
        lookup_policy,
        update_ticket_to_new_flight,
        cancel_ticket,
        check_flight_for_upgrade_space,
    ]

    return create_react_agent(
        llm,
        tools=tools,
        prompt=flight_agent_prompt,
        state_schema=State,
        name=name,
    )
