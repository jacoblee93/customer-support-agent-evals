from typing import Optional

from datetime import datetime

from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_core.prompts import ChatPromptTemplate

from customer_support.agents.state import State

from customer_support.agents.tools import (
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
)


def initialize_hotel_agent(
    llm: BaseChatModel,
    additional_tools: list,
    name: str,
    test_date: Optional[datetime] = None,
):
    SYSTEM_PROMPT = """
You are a specialized assistant for handling hotel bookings. 
The primary assistant delegates work to you whenever the user needs help booking a hotel. 
Search for available hotels based on the user's preferences and confirm the booking details with the customer. 
When searching, be persistent. Expand your query bounds if the first search returns no results. 
If you need more information or the customer changes their mind, escalate the task back to the main assistant.
Remember that a booking isn't completed until after the relevant tool has successfully been used.
Do not waste the user's time. Do not make up invalid tools or functions.

Current user:
<User>
{user_info}
</User>

Current time:
{time}
"""

    hotel_agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=test_date if test_date else datetime.now())

    tools = [
        *additional_tools,
        search_hotels,
        book_hotel,
        update_hotel,
        cancel_hotel,
    ]

    return create_react_agent(
        llm,
        tools=tools,
        prompt=hotel_agent_prompt,
        state_schema=State,
        name=name,
    )
