from dataclasses import dataclass
from typing import List

@dataclass
class PlayerGuess:
    word: str
    result: str
    
    def to_dict(self):
        return {
            "word": self.word,
            "result": self.result,
        }

@dataclass
class MasterStateMessage:
    team_words: list
    opponent_words: list
    neutral_words: list
    guessed_words_log: list[PlayerGuess]

    def to_dict(self):
        return {
            "team_words": self.team_words,
            "opponent_words": self.opponent_words,
            "neutral_words": self.neutral_words,
            "guessed_words_log": [
                g.to_dict() if hasattr(g, "to_dict") else g
                for g in self.guessed_words_log
            ],
        }
    
@dataclass
class PlayerStateMessage:
    hint_word: str
    hint_number: int
    board: list
    guessed_words_log: list[PlayerGuess]

    def to_dict(self):
        return {
            "hint": {
                "word": self.hint_word,
                "number": self.hint_number
            },
            "board": self.board,
            "guessed_words_log": [
                g.to_dict() if hasattr(g, "to_dict") else g
                for g in self.guessed_words_log
            ],
        }
    
@dataclass
class MasterActionMessage:
    hint_word: str
    hint_number: int

    def to_dict(self):
        return {
            "hint_word": self.hint_word,
            "hint_number": self.hint_number
        }
        
@dataclass
class PlayerActionMessage:
    guesses: list

    def to_dict(self):
        return {
            "guesses": self.guesses
        }