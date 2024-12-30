import os
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()

model ="gpt-3.5-turbo"

llm_config = {
    "model": model,
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "temperature": 0.9, 
}

travl_agent=ConversableAgent(
    name="TravelAssistant",
    system_message="You are a helpful AI traveler planning a vacation",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

guide_agent = ConversableAgent(
    name="guide_agent",
    system_message="You are a helpful AI travel guide with extensive knowledge about popular destinations",
)

chat_result = travl_agent.initiate_chat(
    guide_agent, 
    message="What is the must-see attraction in New York?",
    summary_method="reflection_with_llm",
    max_turns=3,
    )

# print("\n************************\n")
# print(chat_result)
# print("\n************************\n")
# print(chat_result.summary)

import pprint
print("\n************************\n")
pprint.pprint(chat_result.chat_history)


print("\n************************\n")
pprint.pprint(chat_result.cost)


