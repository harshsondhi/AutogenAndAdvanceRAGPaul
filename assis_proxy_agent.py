import os
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from dotenv import load_dotenv


load_dotenv()

model = "gpt-3.5-turbo"

llm_config = {
    "model": model,
    "api_key": os.getenv("OPENAI_API_KEY"),
}

assistant = AssistantAgent(
    name="HarshAssistantagent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER"
)

user_proxy = UserProxyAgent(
    name="HarshUserproxyagent",
    llm_config=llm_config,
    code_execution_config={
        "workd_dir": "code_execution",
        "use_docker": False,},
    human_input_mode="NEVER"
)

user_proxy.initiate_chat(
    assistant,
    message="What is the capital of france?"
)