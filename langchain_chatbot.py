from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Template for how the LLM will format and answer questions
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
# Uses langchain to chain together the prompt to the model
chain = prompt | model

def handle_conversation():
    """Simple AI Chatbot"""
    context = ""
    print("Welcome to Llama 3.2 Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        # Passes context and question variables to the template using langchain
        result = chain.invoke({"context": context, "question": user_input})
        print("AI: ", result)
        # Saves results and user_input to context
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handle_conversation()