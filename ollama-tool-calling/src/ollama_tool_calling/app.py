import ollama
from langchain.agents import create_tool_calling_agent, tool, AgentExecutor
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def get_sum(number1: int, number2: int) -> int:
    """Returns sum of two numbers

    Args:
        number1 (int): Number 1
        number2 (int): Number 2

    Returns:
        int: Sum of two numbers
    """
    return number1 + number2

def main():
    prompt = ChatPromptTemplate.from_messages([("system","""
                                                You are a human assitant to add two numbers.
                                                """),
                                               ("human","What is the sum of 1999999999 and 320332930293"),MessagesPlaceholder("agent_scratchpad")])
    model = ChatOllama(model="llama3.2", temperature=0, seed=1)
    agent = create_tool_calling_agent(model, tools=[get_sum],prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[get_sum], verbose=False)
    response = agent_executor.invoke({})
    print(response['output'])


if __name__ == "__main__":
    main()