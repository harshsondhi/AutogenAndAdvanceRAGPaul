import os
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from dotenv import load_dotenv
load_dotenv()

model = "gpt-3.5-turbo"

llm_config = {
    "model": model,
    "api_key": os.environ.get("OPENAI_API_KEY"),
}

human_proxy = UserProxyAgent(
    name="human_proxy",
    human_input_mode="ALWAYS",
    code_execution_config=False,
    llm_config=False
)

agent_with_animal = ConversableAgent(
    "agent_with_animal",
    system_message="You are thinking of an animal. You have the animal 'elephant' in your mind, and I will try to guess it. "
    "If I guess incorrectly, give me a hint. ",
    llm_config=llm_config,
    is_termination_msg=lambda msg: "elephant" in msg["content"],  # terminate if the animal is guessed
    human_input_mode="NEVER",  # never ask for human input
)

human_proxy.initiate_chat(
    agent_with_animal,
    message="Parrot"
)
