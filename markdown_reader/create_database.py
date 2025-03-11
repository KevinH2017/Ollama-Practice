from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

DATA_PATH = "./markdown_reader/books"
EMBEDDING = "nomic-embed-text"
VECTOR_STORE = "simple-rag"
CHROMA_PATH = "./markdown_reader/chroma_db"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    logging.info("Loaded documents")
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info("Text split")
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    embedding = OllamaEmbeddings(model=EMBEDDING)
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding, 
        collection_name=VECTOR_STORE,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    logging.info("DB created")
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()