import os
import json
import random
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class GameMode(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class PromptType(Enum):
    SYSTEM = "system"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_BASED = "role_based"
    TEMPLATE = "template"

@dataclass
class PromptConfig:
    """Configuration for prompt engineering parameters"""
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

@dataclass
class GameState:
    """Track game state and configuration"""
    attempts: int = 0
    max_attempts: int = 4
    game_over: bool = False
    game_mode: GameMode = GameMode.MEDIUM
    target_word: str = ""
    hints_given: int = 0
    max_hints: int = 3

class PromptEngineer:
    """Advanced prompt engineering class with multiple techniques"""
    
    def __init__(self):
        self.config = PromptConfig()
        self.game_state = GameState()
        self.conversation_history = []
        
    def get_system_prompts(self) -> Dict[str, str]:
        """Role-based system prompts for different scenarios"""
        return {
            "game_master": """You are an expert word guessing game master with 20+ years of experience in educational games. 
            Your role is to create engaging, educational, and fun word guessing experiences.
            
            PERSONALITY TRAITS:
            - Enthusiastic and encouraging
            - Patient with learners
            - Creative in clue generation
            - Educational focus
            
            EXPERTISE AREAS:
            - Vocabulary building
            - Critical thinking development
            - Pattern recognition
            - Language learning techniques""",
            
            "hint_generator": """You are a specialized hint generation AI with expertise in:
            - Progressive difficulty scaling
            - Cognitive load management
            - Learning psychology
            - Word association patterns
            
            Your role is to provide hints that guide without giving away the answer.""",
            
            "game_analyst": """You are a game analytics AI that evaluates:
            - Player performance patterns
            - Difficulty progression
            - Learning outcomes
            - Engagement metrics
            
            Provide insights to improve the gaming experience."""
        }
    
    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Few-shot learning examples for different game scenarios"""
        return [
            {
                "scenario": "initial_clue",
                "examples": [
                    {
                        "word": "ELEPHANT",
                        "clue": "I'm a large mammal with a long trunk and big ears. I'm known for my memory and live in Africa and Asia.",
                        "difficulty": "easy"
                    },
                    {
                        "word": "SERENDIPITY",
                        "clue": "I'm a noun that describes the pleasant surprise of finding something valuable when you weren't looking for it.",
                        "difficulty": "hard"
                    }
                ]
            },
            {
                "scenario": "hint_progression",
                "examples": [
                    {
                        "word": "PHOTOSYNTHESIS",
                        "hints": [
                            "I'm a biological process that plants use to make food.",
                            "I require sunlight, water, and carbon dioxide.",
                            "I produce glucose and oxygen as byproducts.",
                            "I happen in the chloroplasts of plant cells."
                        ]
                    }
                ]
            },
            {
                "scenario": "encouragement",
                "examples": [
                    {
                        "situation": "wrong_guess",
                        "response": "That's a good try! You're thinking in the right direction. The word I'm thinking of is related to your guess but has a different meaning. Would you like a hint to help narrow it down?"
                    },
                    {
                        "situation": "close_guess",
                        "response": "You're getting warmer! That's very close to the right answer. Think about the specific context I mentioned in my clue."
                    }
                ]
            }
        ]
    
    def create_chain_of_thought_prompt(self, context: str) -> str:
        """Create chain-of-thought reasoning prompts"""
        return f"""
        Let's think step by step about this word guessing scenario:
        
        CONTEXT: {context}
        
        REASONING PROCESS:
        1. First, I need to analyze what the player is trying to guess
        2. Then, I should consider their current understanding level
        3. Next, I'll determine the appropriate difficulty of my response
        4. Finally, I'll craft a response that guides without giving away the answer
        
        STEP-BY-STEP ANALYSIS:
        - Current attempt number: {self.game_state.attempts}
        - Remaining attempts: {self.game_state.max_attempts - self.game_state.attempts}
        - Hints given so far: {self.game_state.hints_given}
        - Player's last guess: {context}
        
        REASONING: Based on this analysis, I should...
        """
    
    def create_template_prompts(self) -> Dict[str, str]:
        """Template-based prompts for different game phases"""
        return {
            "game_start": """
            🎮 WELCOME TO THE ULTIMATE WORD GUESSING CHALLENGE! 🎮
            
            GAME RULES:
            - You have {max_attempts} attempts to guess the word
            - I'll give you an initial clue
            - You can ask for hints (up to {max_hints} total)
            - Each hint will be more specific than the last
            - Think carefully and use your reasoning skills!
            
            DIFFICULTY LEVEL: {difficulty}
            CATEGORY: {category}
            
            Ready to test your vocabulary and reasoning skills? Let's begin!
            """,
            
            "hint_request": """
            HINT REQUEST ANALYSIS:
            - Current attempt: {attempts}/{max_attempts}
            - Previous hints given: {hints_given}
            - Player's last guess: "{last_guess}"
            
            HINT STRATEGY:
            - Provide a hint that's more specific than the last
            - Guide toward the answer without revealing it
            - Consider the player's learning style
            - Maintain engagement and challenge
            """,
            
            "game_end": """
            GAME COMPLETION ANALYSIS:
            - Final attempt: {attempts}/{max_attempts}
            - Total hints used: {hints_given}
            - Success: {success}
            
            FEEDBACK STRATEGY:
            - Congratulate on success or encourage for next time
            - Provide educational insights about the word
            - Suggest related vocabulary to explore
            - Maintain positive learning environment
            """
        }
    
    def build_conversation_context(self, user_input: str, prompt_type: PromptType) -> List[Dict[str, str]]:
        """Build comprehensive conversation context with prompt engineering"""
        messages = []
        
        # Add role-based system prompt
        system_prompt = self.get_system_prompts()["game_master"]
        messages.append({"role": "system", "content": system_prompt})
        
        # Add few-shot examples based on context
        if "hint" in user_input.lower() or self.game_state.hints_given > 0:
            few_shot_examples = self.get_few_shot_examples()[1]  # hint_progression
            for example in few_shot_examples["examples"]:
                messages.append({
                    "role": "system", 
                    "content": f"EXAMPLE HINT PROGRESSION: {json.dumps(example, indent=2)}"
                })
        
        # Add chain-of-thought reasoning
        if prompt_type == PromptType.CHAIN_OF_THOUGHT:
            cot_prompt = self.create_chain_of_thought_prompt(user_input)
            messages.append({"role": "system", "content": cot_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def get_ai_response(self, user_input: str, prompt_type: PromptType = PromptType.ROLE_BASED, 
                       stream: bool = True, custom_config: Optional[PromptConfig] = None) -> str:
        """Get AI response with advanced prompt engineering"""
        
        # Use custom config or default
        config = custom_config or self.config
        
        # Build conversation context
        messages = self.build_conversation_context(user_input, prompt_type)
        
        # Create completion with all parameters
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            stop=config.stop,
            stream=stream,
        )
        
        if stream:
            ai_response = ""
            print("\n🤖 AI: ", end='', flush=True)
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end='', flush=True)
                    ai_response += content
            print()  # New line after streaming
        else:
            ai_response = response.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        return ai_response
    
    def adjust_parameters_for_difficulty(self, difficulty: GameMode) -> PromptConfig:
        """Adjust prompt parameters based on game difficulty"""
        configs = {
            GameMode.EASY: PromptConfig(
                max_tokens=800,
                temperature=0.5,
                top_p=0.8,
                frequency_penalty=0.1
            ),
            GameMode.MEDIUM: PromptConfig(
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0
            ),
            GameMode.HARD: PromptConfig(
                max_tokens=1200,
                temperature=0.8,
                top_p=0.95,
                frequency_penalty=-0.1
            ),
            GameMode.EXPERT: PromptConfig(
                max_tokens=1500,
                temperature=0.9,
                top_p=0.98,
                frequency_penalty=-0.2
            )
        }
        return configs.get(difficulty, self.config)
    
    def create_meta_prompt(self, task: str) -> str:
        """Create meta-prompting for self-reflection and improvement"""
        return f"""
        META-PROMPT FOR SELF-ANALYSIS:
        
        TASK: {task}
        
        SELF-REFLECTION QUESTIONS:
        1. What is the core objective of this interaction?
        2. What prompt engineering techniques am I using?
        3. How can I improve the user experience?
        4. What learning outcomes am I facilitating?
        5. How can I make this more engaging?
        
        IMPROVEMENT STRATEGIES:
        - Adjust temperature for more/less creativity
        - Modify top_p for more/less focused responses
        - Use chain-of-thought for complex reasoning
        - Apply few-shot learning for consistency
        - Implement role-based prompting for personality
        
        CURRENT CONFIGURATION:
        - Max tokens: {self.config.max_tokens}
        - Temperature: {self.config.temperature}
        - Top-p: {self.config.top_p}
        - Game mode: {self.game_state.game_mode.value}
        """

# Initialize the prompt engineer
prompt_engineer = PromptEngineer()

def get_ai_response(messages, stream=True):
    """Legacy function for backward compatibility"""
    return prompt_engineer.get_ai_response(messages[-1]["content"], stream=stream)

def display_prompt_engineering_info():
    """Display information about prompt engineering techniques being used"""
    print("\n🧠 PROMPT ENGINEERING TECHNIQUES IN USE:")
    print("=" * 50)
    print("✅ Role-based prompting (Game Master persona)")
    print("✅ Few-shot learning examples")
    print("✅ Chain-of-thought reasoning")
    print("✅ Template-based prompts")
    print("✅ Meta-prompting for self-reflection")
    print("✅ Dynamic parameter adjustment")
    print("✅ Context-aware conversation building")
    print("✅ Progressive difficulty scaling")
    print("=" * 50)

def select_difficulty():
    """Let user select game difficulty"""
    print("\n🎯 SELECT DIFFICULTY LEVEL:")
    print("1. Easy (Temperature: 0.5, Max Tokens: 800)")
    print("2. Medium (Temperature: 0.7, Max Tokens: 1000)")
    print("3. Hard (Temperature: 0.8, Max Tokens: 1200)")
    print("4. Expert (Temperature: 0.9, Max Tokens: 1500)")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice == "1":
            return GameMode.EASY
        elif choice == "2":
            return GameMode.MEDIUM
        elif choice == "3":
            return GameMode.HARD
        elif choice == "4":
            return GameMode.EXPERT
        else:
            print("Please enter a valid choice (1-4)")

def demonstrate_prompt_techniques():
    """Demonstrate different prompt engineering techniques"""
    print("\n🔬 PROMPT ENGINEERING DEMONSTRATION:")
    print("=" * 50)
    
    # Demonstrate chain-of-thought prompting
    print("\n1. CHAIN-OF-THOUGHT PROMPTING:")
    cot_response = prompt_engineer.get_ai_response(
        "I'm thinking of a word that starts with 'S' and is related to science",
        PromptType.CHAIN_OF_THOUGHT,
        stream=False
    )
    
    # Demonstrate few-shot learning
    print("\n2. FEW-SHOT LEARNING:")
    few_shot_response = prompt_engineer.get_ai_response(
        "Give me a clue for a word related to space",
        PromptType.FEW_SHOT,
        stream=False
    )
    
    # Demonstrate role-based prompting
    print("\n3. ROLE-BASED PROMPTING:")
    role_response = prompt_engineer.get_ai_response(
        "What makes a good word guessing game?",
        PromptType.ROLE_BASED,
        stream=False
    )

def main_game_loop():
    """Main game loop with advanced prompt engineering"""
    # Display prompt engineering info
    display_prompt_engineering_info()
    
    # Select difficulty
    difficulty = select_difficulty()
    prompt_engineer.game_state.game_mode = difficulty
    
    # Adjust parameters for difficulty
    config = prompt_engineer.adjust_parameters_for_difficulty(difficulty)
    prompt_engineer.config = config
    
    print(f"\n🎮 Starting {difficulty.value.upper()} mode game!")
    print(f"📊 Configuration: Max Tokens={config.max_tokens}, Temperature={config.temperature}, Top-p={config.top_p}")
    
    # Get initial clue using template prompting
    template_prompts = prompt_engineer.create_template_prompts()
    game_start_prompt = template_prompts["game_start"].format(
        max_attempts=prompt_engineer.game_state.max_attempts,
        max_hints=prompt_engineer.game_state.max_hints,
        difficulty=difficulty.value.upper(),
        category="General Knowledge"
    )
    
    print("\n" + "=" * 50)
    print(game_start_prompt)
    print("=" * 50)
    
    # Get initial clue from AI with role-based prompting
    initial_prompt = "Let's play a word guessing game! Give me a word to guess with an initial clue."
    ai_response = prompt_engineer.get_ai_response(
        initial_prompt, 
        PromptType.ROLE_BASED,
        custom_config=config
    )
    
    # Game loop with advanced prompt engineering
    while (prompt_engineer.game_state.attempts < prompt_engineer.game_state.max_attempts and 
           not prompt_engineer.game_state.game_over):
        
        prompt_engineer.game_state.attempts += 1
        print(f"\n📝 Attempt {prompt_engineer.game_state.attempts}/{prompt_engineer.game_state.max_attempts}")
        
        # Get user input
        user_guess = input("Your guess: ").strip()
        
        if not user_guess:
            print("Please enter a guess!")
            prompt_engineer.game_state.attempts -= 1
            continue
        
        # Use chain-of-thought for analyzing the guess
        guess_analysis = f"Player guessed: '{user_guess}'. Analyze this guess and provide feedback."
        ai_response = prompt_engineer.get_ai_response(
            guess_analysis,
            PromptType.CHAIN_OF_THOUGHT,
            custom_config=config
        )
        
        # Check if game should continue
        if any(word in ai_response.lower() for word in ["congratulations", "correct", "you got it", "right", "exactly"]):
            prompt_engineer.game_state.game_over = True
            print("\n🎉 Game Complete! Thanks for playing! 🎉")
            
            # Use meta-prompting for final analysis
            meta_analysis = prompt_engineer.create_meta_prompt("Game completion analysis")
            print(f"\n🧠 {meta_analysis}")
            break
        
        # If not game over and not last attempt, ask about hints
        if (prompt_engineer.game_state.attempts < prompt_engineer.game_state.max_attempts and 
            not prompt_engineer.game_state.game_over and
            prompt_engineer.game_state.hints_given < prompt_engineer.game_state.max_hints):
            
            hint_choice = input("\n💡 Would you like a hint? (y/n): ").strip().lower()
            if hint_choice in ['y', 'yes']:
                prompt_engineer.game_state.hints_given += 1
                
                # Use template prompting for hint request
                hint_template = prompt_engineer.create_template_prompts()["hint_request"].format(
                    attempts=prompt_engineer.game_state.attempts,
                    max_attempts=prompt_engineer.game_state.max_attempts,
                    hints_given=prompt_engineer.game_state.hints_given,
                    last_guess=user_guess
                )
                
                hint_request = f"{hint_template}\n\nProvide a helpful hint for the word."
                hint_response = prompt_engineer.get_ai_response(
                    hint_request,
                    PromptType.FEW_SHOT,
                    custom_config=config
                )
    
    # Game over message if max attempts reached
    if (prompt_engineer.game_state.attempts >= prompt_engineer.game_state.max_attempts and 
        not prompt_engineer.game_state.game_over):
        print(f"\n😔 Game Over! You've used all {prompt_engineer.game_state.max_attempts} attempts.")
        print("Thanks for playing! 🎮")
        
        # Final meta-analysis
        final_analysis = prompt_engineer.create_meta_prompt("Game over analysis")
        print(f"\n🧠 {final_analysis}")
    
    print("\n" + "=" * 50)
    print("🎓 PROMPT ENGINEERING TECHNIQUES USED IN THIS SESSION:")
    print("=" * 50)
    print("✅ Role-based prompting with Game Master persona")
    print("✅ Few-shot learning with example scenarios")
    print("✅ Chain-of-thought reasoning for complex analysis")
    print("✅ Template-based prompts for consistent formatting")
    print("✅ Meta-prompting for self-reflection")
    print("✅ Dynamic parameter adjustment based on difficulty")
    print("✅ Context-aware conversation building")
    print("✅ Progressive hint generation")
    print("✅ Temperature control for creativity (0.5-0.9)")
    print("✅ Top-p sampling for response diversity (0.8-0.98)")
    print("✅ Max tokens optimization (800-1500)")
    print("✅ Frequency and presence penalties")
    print("=" * 50)

# Start the enhanced game
if __name__ == "__main__":
    main_game_loop()