from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

def main():
    file_path = "/Users/fagom/Downloads/Product_Management_Wiki.pdf"
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    embeddings = OllamaEmbeddings(model="llama3.2")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,chunk_overlap=200,add_start_index=True
    )
    
    all_splits = text_splitter.split_documents(docs)
    
    ids = vector_store.add_documents(documents=all_splits)
    
    results = vector_store.similarity_search_with_score("Why Reading Product Management Books is Essential")
    
    doc, score = results[0]
    print(f"Score: {score}\n")
    print(doc.page_content)
    
if __name__ == "__main__":
    main()