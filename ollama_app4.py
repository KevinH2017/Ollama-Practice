# PDF RAG app
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama, os, logging

# DOC_PATH = "./data/BOI.pdf"
DOC_PATH = "./data/SRT-01_Stops_Rust_Enamel_Sprays_TDS.pdf"
MODEL = "llama3.2"
EMBEDDING = "nomic-embed-text"
VECTOR_STORE = "simple-rag"


def doc_loader(doc_path):
    """Loads and returns document for processing"""
    if os.path.exists(doc_path):
        logging.info("Loading...")
        loader = PyPDFLoader(file_path=doc_path)
        data = loader.load_and_split()
        return data
    else:
        logging.error("PDF Upload Failed")
        return None


def split_chunk_doc(doc):
    """Splits document into chunks and returns them"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(doc)
    logging.info("Splitting completed!")
    return chunks


def create_vector_db(chunks):
    """Creates and returns vector database with ollama model"""
    ollama.pull(EMBEDDING)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING),
        collection_name=VECTOR_STORE,
    )
    logging.info("Successfully added to Vector Database!")
    return vector_db


def ollama_retriever(vector_db, llm):
    """Returns relevant chunks of data from vector database based on user's query"""
    # Sets up ollama model
    llm = ChatOllama(model=MODEL)
    # Generates multiple questions from a single question and then retrieves documents based on the questions
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
    # Load and process the PDF document
    data = doc_loader(DOC_PATH)
    if data is None:
        return

    # Splits the documents into chunks
    chunks = split_chunk_doc(data)

    # Creates the vector database
    vector_db = create_vector_db(chunks)

    # Initializes the language model
    llm = ChatOllama(model=MODEL)

    # Creates the retriever
    retriever = ollama_retriever(vector_db, llm)

    # Creates the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    # question = "How to report BOI?"
    question = "Give me instructions on how to use the product."

    # Get and print response
    res = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()