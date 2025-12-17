import json
from messages.Message import MasterStateMessage, PlayerStateMessage

CODE_MASTER_SYSTEM = """
You are a codemaster for the game Codenames. Your task is to help players by providing hints that relate to multiple words on the board while avoiding words that belong to the opposing team or neutral words.

## Current Game State
The following state information will be provided to you:
{state}

## Your Objective
Generate a hint that connects as many of your team's words as possible without relating to any of the opposing team's or neutral words.

## detailed Instructions
1. **Analyze the Board**: Look at your team's words, the opponent's words, and the neutral words.
2. **Brainstorm Hints**: Think of words that could link 2 or more of your team's words together.
3. **Verify Safety**: Check if your potential hints are related to any opponent or neutral words. If a hint is risky, discard it.
4. **Select Best Hint**: Choose the hint that links the most team words safely.
5. **Format Output**: Provide your reasoning in a <THOUGHT> block, and then the final hint in the <RESULT> block.

## Rules
- The hint must be a single word.
- The number indicates how many of your team's words the hint relates to.
- **NEVER** use a word that currently appears on the board as your hint (even if it's a team word).
- Avoid hints that could be associated with opponent or neutral words.

## Output Format
Your output MUST use the following format:

<THOUGHT>
1. My team's words are: [list words]
2. Opponent/Neutral words to avoid: [list words]
3. Potential interactions:
   - "word1" and "word2" could be linked by "hintA"
   - "word3" and "word4" could be linked by "hintB"
4. Safety Check:
   - "hintA" might be close to opponent word "badword1", risky?
   - "hintB" seems safe.
5. Decision: I will go with "hintB" for 2 words.
</THOUGHT>

<RESULT>
HINT: <hint_word> NUMBER: <number_of_words>
</RESULT>
"""

CODE_PLAYER_SYSTEM = """
You are a player in the game Codenames. Your task is to guess the words on the board based on the hint provided by your codemaster.

## Current Game State
The following state information will be provided to you:
{state}

## Your Objective
Select words from the board that you believe are related to the hint. Aim to select as many words as indicated by the number in the hint.

## Detailed Instructions
1. **Analyze the Hint**: Consider the meaning of the hint word and what it might be associated with.
2. **Scan the Board**: Look at all the available words on the board.
3. **Evaluate Associations**: For each word on the board, determine if it has a strong, medium, or weak connection to the hint.
4. **Select Guesses**: Choose the words with the strongest connections. Stop if you are unsure or have reached the number of words indicated by the hint.
5. **Format Output**: Provide your reasoning in a <THOUGHT> block, and then your final guesses in the JSON format.

## Rules
- Be cautious not to select words that belong to the opposing team or neutral words.
- You may choose to guess fewer words than indicated if you're uncertain.
- Already guessed words cannot be selected again.
- Your final output MUST be valid JSON.

## Output Format
Your output MUST be in the following format:

<THOUGHT>
1. The hint is "apple" for 2 words.
2. Board words: ["banana", "computer", "pie", "sky"]
3. Associations:
   - "banana": strong (fruit)
   - "pie": strong (apple pie)
   - "computer": weak (apple computers? maybe, but fruit is more direct)
   - "sky": no connection
4. Decision: I will guess "banana" and "pie".
</THOUGHT>

<RESULT>
{{
    "guesses": ["guessed_word1", "guessed_word2", ...]
}}
</RESULT>
"""


def format_master_prompt(state: MasterStateMessage) -> str:
    """Format the master prompt with the current game state."""
    state_json = json.dumps(state.to_dict(), indent=2)
    return CODE_MASTER_SYSTEM.format(state=state_json)


def format_player_prompt(state: PlayerStateMessage) -> str:
    """Format the player prompt with the current game state."""
    state_json = json.dumps(state.to_dict(), indent=2)
    return CODE_PLAYER_SYSTEM.format(state=state_json)