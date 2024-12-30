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

def add_numbers(
        a: Annotated[int, "First number"], b: Annotated[int, "Second number"]
    )-> str:
    return f"The sum of {a} and {b} is {a + b}."


def multiply_numbers(
        a: Annotated[int, "First number"], b: Annotated[int, "Second number"]
    )-> str:
    return f"The product of {a} and {b} is {a * b}."

assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    code_execution_config=False,
    system_message="You are a helpful AI calculator, return 'TERMINATE' when the task is done.",
    human_input_mode="NEVER",
)

user_proxy= ConversableAgent(
    name="user",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

assistant.register_for_llm(
    name="add_numbers",
    description="Add two numbers",
)(add_numbers)

assistant.register_for_llm(
    name="multiply_numbers",
    description="Multiply two numbers",
)(multiply_numbers)

user_proxy.register_for_execution(name="add_numbers")(add_numbers)
user_proxy.register_for_execution(name="multiply_numbers")(multiply_numbers)

user_proxy.initiate_chat(assistant, message="What is the sum of 7 and 5?")