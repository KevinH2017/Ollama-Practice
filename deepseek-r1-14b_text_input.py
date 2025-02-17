import ollama

content = str(input("Ask deepseek-r1:14b: ") + " Show your thought process")

stream = ollama.chat(
    model="deepseek-r1:14b",
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