import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a word guessing game master. Create a fun guessing game where you describe a word, object, animal, or concept using clues, and the user needs to guess what it is. Give hints progressively - start with general clues and get more specific if they need help. When the user guesses correctly, congratulate them enthusiastically! When they guess incorrectly, encourage them to try again and provide additional hints. Make it engaging and educational!"},
        {"role": "user", "content": "Let's play a guessing game! Give me a word to guess."}
    ],
    stream=True,
)

print("Streaming response:")
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end='', flush=True)
print("\n\nStreaming complete!")