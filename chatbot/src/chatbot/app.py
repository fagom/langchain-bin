from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
model = init_chat_model(model="llama3.2",model_provider='ollama')

# Define the function that calls the model
def call_model(state:MessagesState):
    response = model.invoke(state['messages'])
    return {"messages":response}

workflow.add_edge(START, "model")
workflow.add_node("model",call_model)

def main():
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}} # the thread_id is used for memnory purposes
    
    query = input("> ")
    while(query != "bye"):
        input_messages = [HumanMessage(query)]
        #output = app.invoke({"messages": input_messages}, config)
        #output["messages"][-1].pretty_print()  # output contains all messages in state
        
        # the below is used for streaming the response
        for chunk, metadata in app.stream(
            {"messages":input_messages, "language":"English"},
            config=config,
            stream_mode="messages"):
            if isinstance(chunk, AIMessage):
                print(chunk.content, end="")
        print("\n")
        query = input("> ")
    

if __name__ == "__main__":
    main()