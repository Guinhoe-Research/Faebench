from messages.Message import MasterActionMessage, PlayerActionMessage
from typing import Any, List, Tuple
import random

class Environment:
    def __init__(self, config):
        # Game state
        
        if isinstance(config, dict):
            self.teams = config.get("teams", 1)
            self.max_words = config.get("max_words", 25)
            word_list_file = config.get("word_list_file", None)
        else:
            self.teams = config.teams
            self.max_words = config.max_words
            word_list_file = config.word_list_file
            
        self.word_sets = {}
        self.neutral_words = []
        self.board = []

        self.guessed_words = []
        self.guessed_words_log = {i: [] for i in range(1, self.teams + 1)}

        # Configuration parameters
        if word_list_file:
            with open(word_list_file, 'r') as f:
                word_list = [line.strip() for line in f.readlines()]
                self._setup_board(word_list)
        else:
            if "test_flag" in config and config["test_flag"]:
                pass  # Skip loading word list in test mode
            else:
                raise ValueError("No word list provided in config")

    def _setup_board(self, word_list: List[str]):
        random.shuffle(word_list)
        self.board = word_list[:self.max_words]
        if self.teams > 1:
            # Setup for 2 teams
            board_copy = self.board.copy()
            random.shuffle(board_copy)
            self.word_sets[1] = board_copy[:8]
            self.word_sets[2] = board_copy[8:16]
            self.neutral_words = [word for word in self.board if word not in self.word_sets[1] and word not in self.word_sets[2]]
            # Additional setup for multiple teams can be added here
        else:
            board_copy = self.board.copy()
            random.shuffle(board_copy)
            self.word_sets[1] = board_copy[:8]
            self.neutral_words = [word for word in self.board if word not in self.word_sets[1]]
        
    def check_win(self) -> bool:
        if self.teams > 1:
            # Check win conditions for multiple teams
            for team, words in self.word_sets.items():
                remaining = [w for w in words if w not in self.guessed_words]
                if len(remaining) == 0:
                    return True
        else:
            # Win if all team words have been guessed
            remaining = [w for w in self.word_sets[1] if w not in self.guessed_words]
            return len(remaining) == 0
        return False

    def get_winner(self) -> int:
        if self.teams > 1:
            for team, words in self.word_sets.items():
                remaining = [w for w in words if w not in self.guessed_words]
                if len(remaining) == 0:
                    return team
        else:
            remaining = [w for w in self.word_sets[1] if w not in self.guessed_words]
            if len(remaining) == 0:
                return 1
        return -1
    
    def get_master_state(self, team_id: int = 1) -> dict:
        if team_id not in self.word_sets:
            # Ensure safe fallback for backward compatibility if team_id depends on incomplete init
            if self.teams == 1 and team_id == 1 and 1 not in self.word_sets:
                 # Should have been initialized, but if manually setting up board...
                 pass
            else:
                 raise ValueError(f"Invalid team id {team_id}. Available: {list(self.word_sets.keys())}")
        
        return {
            "board": self.board,
            "word_sets": self.word_sets, 
            "guessed_words": self.guessed_words,
            "guessed_words_log": self.guessed_words_log.get(team_id, [])
        }
    
    def get_player_state(self, team_id: int = 1) -> dict:
        """Return the current game state for a player."""
        return {
            "board": [w for w in self.board if w not in self.guessed_words],
            "guessed_words_log": self.guessed_words_log.get(team_id, [])
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
    
    def handle_player_action(self, action: PlayerActionMessage, team_id: int = 1) -> dict:
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
            is_correct = guess in self.word_sets.get(team_id, [])
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
            self.guessed_words_log[team_id].append(guess_dict)
        
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