from messages.Message import MasterActionMessage, PlayerActionMessage
from typing import Any, List
import random

class Environment:
    def __init__(self, config):
        # Game state
        self.teams = config.get("teams", 1)
        self.word_sets = {}
        self.neutral_words = []
        self.board = []

        self.guessed_words = []
        self.guessed_words_log = []

        # Configuration parameters
        self.max_words = config.get("max_words", 25)

        with open(config["word_list_file"], 'r') as f:
            word_list = [line.strip() for line in f.readlines()]
            self._setup_board(word_list)

    def _setup_board(self, word_list: List[str]):
        random.shuffle(word_list)
        self.board = word_list[:self.max_words]
        if self.teams > 1:
            # Setup for multiple teams
            pass
        else:
            board_copy = self.board.copy()
            random.shuffle(board_copy)
            self.word_sets[1] = board_copy[:8]
            self.neutral_words = [word for word in self.board if word not in self.word_sets[1]]
        
    def check_win(self):
        if self.teams > 1:
            # Check win conditions for multiple teams
            pass
        else:
            # Win if all team words have been guessed
            remaining = [w for w in self.word_sets[1] if w not in self.guessed_words]
            return len(remaining) == 0
        return False
    
    def get_master_state(self):
        return {
            "board": self.board,
            "word_sets": self.word_sets,
            "guessed_words": self.guessed_words,
            "guessed_words_log": self.guessed_words_log
        }
    
    def get_player_state(self):
        """Return the current game state for a player."""
        return {
            "current_hint": self.current_hint,
            "board": [w for w in self.board if w not in self.guessed_words],
            "guessed_words_log": self.guessed_words_log
        }
    
    def handle_master_action(self, action: MasterActionMessage) -> dict:
        """Process master action and store the hint."""
        self.current_hint = {
            "word": action.hint_word,
            "number": action.hint_number
        }
        return {
            "success": True,
            "hint": self.current_hint
        }
    
    def handle_player_action(self, action: PlayerActionMessage) -> dict:
        """Process player guesses and update game state."""
        results = []
        correct_count = 0
        
        for guess in action.guesses:
            if guess in self.guessed_words:
                results.append({"word": guess, "result": "already_guessed"})
                continue
            
            if guess not in self.board:
                results.append({"word": guess, "result": "invalid"})
                continue
            
            result = ""
            
            # Check if the guess is correct (team word)
            is_correct = any(guess in words for words in self.word_sets.values())
            if is_correct:
                correct_count += 1
                result = "correct"
            elif guess in self.neutral_words:
                result = "neutral"
            else:
                result = "opponent"
                
            guess_dict = {"word": guess, "result": result}
            
            results.append(guess_dict)
            self.guessed_words.append(guess)
            self.guessed_words_log.append(guess_dict)
        
        return {
            "success": True,
            "results": results,
            "correct_count": correct_count,
            "game_over": self.check_win()
        }
    
    def get_game_state(self) -> dict:
        """Return the full game state."""
        return {
            "board": self.board,
            "word_sets": self.word_sets,
            "neutral_words": self.neutral_words,
            "guessed_words": self.guessed_words
        }