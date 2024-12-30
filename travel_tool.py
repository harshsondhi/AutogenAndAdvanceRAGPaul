import os
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()

model ="gpt-3.5-turbo"

llm_config = {
    "model": model,
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "temperature": 0.0, 
}

def get_flight_status(
        flight_number: Annotated[str, "Flight number"]
    )-> str:
    dummy_data = {"AA123": "On time", "DL456": "Delayed", "UA789": "Cancelled"}
    return f"The flight {flight_number} is {dummy_data.get(flight_number, 'unknown')}."


def get_travel_advice(
        location: Annotated[str, "location"]
    )-> str:
    dummy_data = {
        "New York": "Top hotel in New York: The Plaza - 5 stars",
        "Los Angeles": "Top hotel in Los Angeles: The Beverly Hills Hotel - 5 stars",
        "Chicago": "Top hotel in Chicago: The Langham - 5 stars",
    }
    return dummy_data.get(location, f"No travel advice available for {location}.")



def get_hotel_info(
        location: Annotated[str, "location"]
    ) -> str:
    dummy_data = {
        "New York": "Top hotel in New York: The Plaza - 5 stars",
        "Los Angeles": "Top hotel in Los Angeles: The Beverly Hills Hotel - 5 stars",
        "Chicago": "Top hotel in Chicago: The Langham - 5 stars",
    }
    return dummy_data.get(location, f"No hotels found in {location}.")



assistant = ConversableAgent(
    name="TravelAssistant",
    system_message="You are a he;pfult AI travel assistant, return 'TERMINATE' when the task is done.",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

user_proxy = ConversableAgent(
    name="User",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

assistant.register_for_llm(
    name="get_flight_status",
    description="Get flight status based on flight number",
)(get_flight_status)

assistant.register_for_llm(
    name="get_hotel_info",
    description="Get information about hotels in a specific location",
)(get_hotel_info)

assistant.register_for_llm(
    name="get_travel_advice",
    description="Get travel advice for a specific location",
)(get_travel_advice)

user_proxy.register_for_execution(name="get_flight_status")(get_flight_status)
user_proxy.register_for_execution(name="get_hotel_info")(get_hotel_info)
user_proxy.register_for_execution(name="get_travel_advice")(get_travel_advice)

user_proxy.initiate_chat(assistant, message="I need help with my travel plans. Can you help me? I am traveling to New York. I need hotel information. Also give me the status of my flight AA123.")

