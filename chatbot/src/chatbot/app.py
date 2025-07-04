from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)
model = init_chat_model(model="llama3.2",model_provider='ollama')
console = Console()

# Define the function that calls the model
def call_model(state:MessagesState):
    response = model.invoke(state['messages'])
    return {"messages":response}

workflow.add_edge(START, "model")
workflow.add_node("model",call_model)

async def main():
    async with AsyncConnectionPool(
        conninfo=f"postgres://llm:llm@localhost:5432/llm",
        max_size=20,
        kwargs={
            "autocommit":True
        }
    ) as pool:
        async with pool.connection() as conn:
            memory = AsyncPostgresSaver(conn=conn)
            await memory.setup()
        
        async with pool.connection() as conn:
            memory = AsyncPostgresSaver(conn=conn)
            app = workflow.compile(checkpointer=memory)
            config = {"configurable": {"thread_id": "abc123"}} # the thread_id is used for memnory purposes
            query = input("> ")
            print("\n")
            while(query != "bye"):
                input_messages = [HumanMessage(query)]
                md = ""
        
                # the below is used for streaming the response
                with console.status("[bold green]Generating response...[/]", spinner="dots"):
                    async for chunk, metadata in app.astream(
                        {"messages":input_messages, "language":"English"},
                        config=config,
                        stream_mode="messages"):
                        if isinstance(chunk, AIMessage):
                            md += chunk.content
                            #print(chunk.content, end="")
                console.print(Markdown(md))
                print("\n")
                query = input("> ")
    

if __name__ == "__main__":
    asyncio.run(main())