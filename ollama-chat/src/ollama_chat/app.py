from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage
from langchain_ollama import ChatOllama

def main():
    template = """
    Question: {question}
    
    Answer: Let's thing step by step.
"""
    prompt = ChatPromptTemplate.from_template(template=template)
    # seed is passed to get the same response across multiple requests for the same prompt.
    # should be a non-negative number.
    # temperature is set to create creativity in the responses
    model = OllamaLLM(model="llama3.2", seed=2, temperature=1)
    chain = prompt | model # | used here is called chaining. Allows to add a template to our prompt with params
    response = chain.invoke({"question": """
                             if i give you an OHLC data, can you identify what kind of candle it is?
                             """})
    print("\n"+response+"\n")
    
    
    llm = ChatOllama(model="llama3.2", temperature=0, seed=1)
    messages = [
    (
        "system",
        "You are a helpful assistant that can turn any text into binary. Convert the text user provides",
    ),
    ("human", "I love programming.")
    ]
    
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
    

if __name__ == "__main__":
    main()