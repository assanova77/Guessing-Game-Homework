import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Initialize game
messages = [
    {"role": "system", "content": "You are a word guessing game master. Create a fun guessing game with these rules: 1) Greet the user warmly and explain the game. 2) Maximum 4 attempts allowed. 3) Give one clue initially, then ask if they want hints after each wrong guess. 4) When they guess incorrectly, encourage them to try differently and ask 'Should I give you hints?' 5) If they say yes to hints, provide progressively more specific clues. 6) When they guess correctly, congratulate them enthusiastically! 7) Track attempts and let them know how many tries they have left. Make it engaging and educational!"},
    {"role": "user", "content": "Let's play a guessing game! Give me a word to guess."}
]

attempts = 0
max_attempts = 4
game_over = False

def get_ai_response(messages, stream=True):
    """Get AI response and handle streaming"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=stream,
    )
    
    if stream:
        ai_response = ""
        print("\nAI: ", end='', flush=True)
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                ai_response += content
        print()  # New line after streaming
        return ai_response
    else:
        return response.choices[0].message.content

# Start the game
print("🎮 Welcome to the AI Word Guessing Game! 🎮")
print("=" * 50)

# Get initial clue from AI
ai_response = get_ai_response(messages)
messages.append({"role": "assistant", "content": ai_response})

# Game loop
while attempts < max_attempts and not game_over:
    attempts += 1
    print(f"\n📝 Attempt {attempts}/{max_attempts}")
    
    # Get user input
    user_guess = input("Your guess: ").strip()
    
    if not user_guess:
        print("Please enter a guess!")
        attempts -= 1  # Don't count empty guesses
        continue
    
    # Add user guess to messages
    messages.append({"role": "user", "content": user_guess})
    
    # Get AI response to the guess
    ai_response = get_ai_response(messages)
    messages.append({"role": "assistant", "content": ai_response})
    
    # Check if game should continue
    if "congratulations" in ai_response.lower() or "correct" in ai_response.lower() or "you got it" in ai_response.lower():
        game_over = True
        print("\n🎉 Game Complete! Thanks for playing! 🎉")
        break
    
    # If not game over and not last attempt, ask about hints
    if attempts < max_attempts and not game_over:
        hint_choice = input("\n💡 Would you like a hint? (y/n): ").strip().lower()
        if hint_choice in ['y', 'yes']:
            messages.append({"role": "user", "content": "yes, give me a hint"})
            hint_response = get_ai_response(messages)
            messages.append({"role": "assistant", "content": hint_response})

# Game over message if max attempts reached
if attempts >= max_attempts and not game_over:
    print(f"\n😔 Game Over! You've used all {max_attempts} attempts.")
    print("Thanks for playing! 🎮")

print("\n" + "=" * 50)