from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(
    model='llama2',
    messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        }],
        stream=True,
)

# Prinst response word by word
for chunk in response:
    print(chunk['message']['content'], end='', flush=True)

# Loads response and then prints as a block of text
# print(response['message']['content'])
# or access fields directly from the response object:
# print(response.message.content)

# Shows the direct message, with ascii and options
# print(response.message)