import json
from datetime import datetime, timedelta
import pytz

test_date = datetime(2025, 3, 20, 14, 0).astimezone(
    pytz.timezone("America/Los_Angeles")
)

policy_check_inputs_trajectory = {
    "messages": [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
                    "function": {
                        "name": "fetch_user_flight_information",
                        "arguments": "{}",
                    },
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "fetch_user_flight_information",
            "tool_call_id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
            "content": json.dumps(
                [
                    {
                        "ticket_no": "7240005432906569",
                        "book_ref": "C46E9F",
                        "flight_id": 19250,
                        "flight_no": "LX0112",
                        "departure_airport": "CDG",
                        "arrival_airport": "BSL",
                        "scheduled_departure": (
                            test_date + timedelta(hours=1)
                        ).isoformat(),
                        "scheduled_arrival": (
                            test_date + timedelta(hours=2)
                        ).isoformat(),
                        "seat_no": "18E",
                        "fare_conditions": "Economy",
                    }
                ]
            ),
        },
        {
            "role": "assistant",
            "content": "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
            "role": "user",
            "content": "Bump me to first class!",
        },
    ]
}

policy_check_reference_trajectory = {
    "messages": policy_check_inputs_trajectory["messages"]
    + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "4a286aff-199a-4152-99b1-df1ca07c920e",
                    "function": {
                        "name": "lookup_policy",
                        "arguments": json.dumps({"query": "flight upgrades"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "lookup_policy",
            "tool_call_id": "4a286aff-199a-4152-99b1-df1ca07c920e",
            "content": "Upgrades to first class are not currently available as they are being saved for VIPs.",
        },
        {
            "role": "assistant",
            "content": "The next flight after that is LX0112 from CDG to BSL is in 4 hours. However, we do not currently allow upgrades to first class. Confirming that I should book it for you anyway?",
        },
    ]
}

handoff_check_inputs_trajectory = {
    "messages": [
        {"role": "user", "content": "Hi there, what time is my flight?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
                    "function": {
                        "name": "fetch_user_flight_information",
                        "arguments": "{}",
                    },
                }
            ],
            "content": "",
        },
        {
            "role": "tool",
            "name": "fetch_user_flight_information",
            "tool_call_id": "d3b6d04c-87b5-4e94-a11f-d8bc7c033188",
            "content": json.dumps(
                [
                    {
                        "ticket_no": "7240005432906569",
                        "book_ref": "C46E9F",
                        "flight_id": 19250,
                        "flight_no": "LX0112",
                        "departure_airport": "CDG",
                        "arrival_airport": "BSL",
                        "scheduled_departure": (
                            test_date + timedelta(hours=1)
                        ).isoformat(),
                        "scheduled_arrival": (
                            test_date + timedelta(hours=2)
                        ).isoformat(),
                        "seat_no": "18E",
                        "fare_conditions": "Economy",
                    }
                ]
            ),
        },
        {
            "role": "assistant",
            "content": "Your flight LX0112 from CDG to BSL is scheduled to depart in an hour and arrive in two hours.",
        },
        {
            "content": "Can you book me a hotel for 3 nights when I land?",
            "role": "user",
        },
    ],
}
