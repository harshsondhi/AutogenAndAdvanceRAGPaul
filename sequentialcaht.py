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

initial_agent=ConversableAgent(
    name="InitialAgent",
    system_message="You return me the text i give you",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

uppercase_agent=ConversableAgent(
    name="UppecaseAgent",
    system_message="You return me the text i give you in uppercase",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

wordcount_agent=ConversableAgent(
    name="WordcountAgent",
    system_message="You count the number of words in the text i give you",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

reverse_text_agent=ConversableAgent(
    name="ReversetextAgent",
    system_message="You reverse the text i give you",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

summarize_agent=ConversableAgent(
    name="SummarizeAgent",
    system_message="You summarize the text i give you",
    human_input_mode="NEVER",
    llm_config=llm_config,
)

chat_results = initial_agent.initiate_chats(
    [
        {
            "recipient": uppercase_agent,
            "message": "Be the change that you wish to see in the world",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": wordcount_agent,
            "message": "These are my numbers",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": reverse_text_agent,
            "message": "These are my numbers",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": summarize_agent,
            "message": "These are my numbers",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
    ]
)

print("First Chat Summary: ", chat_results[0].summary)
print("Second Chat Summary: ", chat_results[1].summary)
print("Third Chat Summary: ", chat_results[2].summary)
print("Fourth Chat Summary: ", chat_results[3].summary)