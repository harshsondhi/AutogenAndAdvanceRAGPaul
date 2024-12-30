import os
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent,GroupChat, GroupChatManager
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()

model ="gpt-3.5-turbo"

llm_config = {
    "model": model,
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "temperature": 0.9, 
}

flight_agent=ConversableAgent(
    name="Flight_Agent",
    system_message="You provide the best flight options for the given destination and dates",
    human_input_mode="NEVER",
    llm_config=llm_config,
    description="""Provides flight option""",
)

hotel_agent=ConversableAgent(
    name="Hotel_Agent",
    system_message="You provide best hotels for the given destination and dates",
    human_input_mode="NEVER",
    llm_config=llm_config,
    description="""Suggest hotel options""",
)

activity_agent=ConversableAgent(
    name="Activity_Agent",
    system_message="You recommend activities and attractions to visit at the destination",
    human_input_mode="NEVER",
    llm_config=llm_config,
    description="""Recommends activities and attractions,""",
)

restaurant_agent=ConversableAgent(
    name="Restaurant_Agent",
    system_message="You suggest the best restaurants to dine at the destination",
    human_input_mode="NEVER",
    llm_config=llm_config,
    description="""Recommends restaurants.""",
)

weather_agent=ConversableAgent(
    name="Weather_Agent",
    system_message="You provide the weather forecast for the travel dates",
    human_input_mode="NEVER",
    llm_config=llm_config,
    description="""Provide weather forcast.""",
)

group_chat = GroupChat(
    agents=[flight_agent, hotel_agent, activity_agent, restaurant_agent, weather_agent],
    messages=[],
    max_round=5,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

chat_result = weather_agent.initiate_chat(
    group_chat_manager,
    message="I'm planning a trip to Paris for the first week of September. Can you help me plan? I will be departuring from Miami",
    summary_method="reflection_with_llm",
)