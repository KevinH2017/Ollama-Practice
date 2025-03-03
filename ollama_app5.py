# PDF RAG app using streamlit
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama, os, logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)

DOC_PATH = "./data/BOI.pdf"
# DOC_PATH = "./data/SRT-01_Stops_Rust_Enamel_Sprays_TDS.pdf"
MODEL = "llama3.2"
EMBEDDING = "nomic-embed-text"
VECTOR_STORE = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"


def doc_loader(doc_path):
    """Loads and returns document for processing"""
    if os.path.exists(doc_path):
        loader = PyPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("Loading document...")
        return data
    else:
        logging.error(f"Document file not found at path: {doc_path}")
        st.error("Document file not found.")
        return None


def split_chunk_doc(documents):
    """Splits document into chunks and returns them"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Splitting completed!")
    return chunks


@st.cache_resource
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
        logging.info("Vector database created and persisted.")
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
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("RAG Chain created successfully!")
    return chain


def main():
    st.title("Document Assistant")

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model
                llm = ChatOllama(model=MODEL)

                # Load the vector database
                vector_db = create_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Create the retriever
                retriever = ollama_retriever(vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()