# Ollama AI Book Chatbot
from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import os, logging, ollama, shutil, sys
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)

DOC_PATH = "./markdown_reader/books"
EMBEDDING = "nomic-embed-text"
VECTOR_STORE = "simple-rag"
PERSIST_DIRECTORY = "./markdown_reader/chroma_db"
MODEL = "llama3.2"
PROMPT_TEMPLATE = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

def doc_loader(doc_path):
    """Loads and returns document for processing"""
    if os.path.exists(doc_path):
        # Gets all *.md files 
        loader = DirectoryLoader(doc_path, glob="*.md")
        data = loader.load()
        logging.info("Loading document...")
        return data
    else:
        logging.error(f"Document file not found at path: {doc_path}")
        logging.error("Document file not found.")
        return None

def split_chunk_doc(documents):
    """Splits document into chunks and returns them"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=100,
        length_function=len, 
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Test chunk
    # document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks

def create_vector_db():
    """Creates and returns vector database with ollama model"""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING)

    embedding = OllamaEmbeddings(model=EMBEDDING)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Successfully added to Vector Database!")
    else:
        # Clear out the database first.
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)

        # Load and process the PDF document
        data = doc_loader(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_chunk_doc(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info(f"Saved {len(chunks)} chunks to {PERSIST_DIRECTORY}.")
    return vector_db

def ollama_retriever(vector_db, llm):
    """Returns relevant chunks of data from vector database based on user's query"""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    # Transforms the db into a retriever to pass questions to the llm using the QUERY_PROMPT 
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Query Retriever successfully created!")
    return retriever

def create_chain(retriever, llm):
    """Passes retriever chunks to llm and returns an answer to the user's query"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("RAG Chain created successfully!")
    return chain

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text

    # query_text = sys.argv[1]

    query_text = "When does Alice meet the Mad Hatter in the book Alice in Wonderland?"

    data = doc_loader(DOC_PATH)
    if data is None:
        return
    
    vector_db = create_vector_db()

    llm = ChatOllama(model=MODEL)
    retriever = ollama_retriever(vector_db, llm)
    logging.info("Creating chain...")
    chain = create_chain(retriever, llm)
    logging.info("Querying text...")
    # Get and print response
    res = chain.invoke(input=query_text)
    
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()