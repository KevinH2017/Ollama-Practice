# PDF RAG app
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# doc_path = "./data/BOI.pdf"
doc_path = "./data/SRT-01_Stops_Rust_Enamel_Sprays_TDS.pdf"
model = "llama3.2"

if doc_path:
    print("Loading...")
    loader = PyPDFLoader(file_path=doc_path)
    data = loader.load_and_split()
else:
    print("PDF Upload Failed")


# Split and chunk PDF
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("Splitting completed!")


# Creates vector databse with ollama model
ollama.pull("nomic-embed-text")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("Completed Adding to Vector Database!")

# Sets up ollama model
llm = ChatOllama(model=model)
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

# res = chain.invoke(input=("What is the document about?",))
# res = chain.invoke(input=("What are the main points as a business owner I should be aware of?",))
# res = chain.invoke(input=("How to report BOI?",))
# res = chain.invoke(input=("Give me a summary about the document.",))
res = chain.invoke(input=("Give me instructions on how to use the product.",))

print(res)