import ollama

stream = ollama.chat(
    model="llama2",
    messages=[
        {
            'role': 'user', 
            'content': 'Why is the sky blue?'
        }],
    stream=True, 
)

# Prints response word by word
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)