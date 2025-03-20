import uuid

from customer_support.agents.swarm import initialize_swarm_agent
# from customer_support.agents.supervisor import initialize_supervisor_agent

from customer_support.db import update_dates, db
from customer_support.utils import print_event
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Update with the backup file so we can restart from the original place in each section
update_dates(db)

thread_id = str(uuid.uuid4())

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

graph = initialize_swarm_agent(llm, MemorySaver())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}


_printed = set()

print("Welcome to the travel assistant! Type 'quit' to exit.")
while True:
    user_input = input("\nWhat can I help you with? > ")

    if user_input.lower() == "quit":
        break

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        print_event(event, _printed)
