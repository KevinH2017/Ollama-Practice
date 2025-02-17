import ollama

content = str(input("Ask llama2: "))

stream = ollama.chat(
    model="llama2",
    messages=[
        {
            'role': 'user', 
            'content': content
        }],
    stream=True, 
)

# Prints response word by word
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)