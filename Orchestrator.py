from abc import ABC, abstractmethod
from typing import Any
import json
import re

import requests

from Environment import Environment
from messages.Message import MasterStateMessage, PlayerStateMessage, MasterActionMessage, PlayerActionMessage
from prompts.agent_prompts import format_master_prompt, format_player_prompt
from Rewards import reward_function


class Orchestrator(ABC):
    def __init__(self, orchestration_config):
        self.environment = Environment(orchestration_config["env_config"])
    
    @abstractmethod
    def get_master_state(self) -> MasterStateMessage:
        pass

    @abstractmethod
    def handle_master_action(self, master_id, action: MasterActionMessage) -> dict:
        pass

    @abstractmethod
    def get_player_state(self, player_id) -> PlayerStateMessage:
        pass

    @abstractmethod
    def handle_player_action(self, player_id, action: PlayerActionMessage) -> dict:
        pass

    def run_full(self) -> dict:
        pass


class OllamaOrchestrator(Orchestrator):
    def __init__(self, config):
        super().__init__(config)
        
        self.master_model = config.get("master_model", "llama3.2b")
        self.player_model = config.get("player_model", "llama3.2b")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434/api/generate")

        self.orchestration_log = {}
        self.reward_log = {}
        self.step_count = 0

    def get_master_state(self) -> MasterStateMessage:
        """Get the current game state formatted for the codemaster."""
        env_state = self.environment.get_master_state()
        
        # Get team words for the specified master
        team_words = env_state["word_sets"].get(1, [])
        
        # Determine opponent words (all other teams' words)
        opponent_words = []
        # for team_id, words in env_state["word_sets"].items():
        #     if team_id != master_id:
        #         opponent_words.extend(words)
        
        # Neutral words are board words not assigned to any team
        all_team_words = set()
        for words in env_state["word_sets"].values():
            all_team_words.update(words)
        neutral_words = [w for w in env_state["board"] if w not in all_team_words]
        
        return MasterStateMessage(
            team_words=team_words,
            opponent_words=opponent_words,
            neutral_words=neutral_words
        )

    def handle_master_action(self, action: MasterActionMessage = None) -> dict:
        """
        Query the master model for a hint and pass it to the environment.
        If action is None, generates a new action from the model.
        Returns the result from the environment.
        """
        result = self.environment.handle_master_action(action)
        
        return {
            "success": True,
            "action": action.to_dict(),
            "environment_result": result
        }

    def get_player_state(self) -> PlayerStateMessage:
        """Get the current game state formatted for a player."""
        env_state = self.environment.get_player_state()
        
        return PlayerStateMessage(
            hint_word=env_state.get("current_hint", {}).get("word", ""),
            hint_number=env_state.get("current_hint", {}).get("number", 0),
            board=env_state.get("board", []),
            guessed_words=env_state.get("guessed_words", [])
        )

    def handle_player_action(self, action: PlayerActionMessage = None) -> dict:
        """
        Query the player model for guesses and pass them to the environment.
        If action is None, generates a new action from the model.
        Returns the result from the environment.
        """
        result = self.environment.handle_player_action(action)
        
        return {
            "success": True,
            "action": action.to_dict(),
            "environment_result": result
        }

    def _query_ollama(self, prompt: str, model: str) -> str:
        """Query the Ollama API with the given prompt and model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            return f"Error querying Ollama: {e}"

    def _parse_master_response(self, response: str) -> MasterActionMessage:
        """Parse the master's response to extract hint word and number."""
        try:
            # Look for the <RESULT>...</RESULT> block
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response
            
            hint_match = re.search(r'HINT:\s*(\w+)\s*NUMBER:\s*(\d+)', result_text, re.IGNORECASE)
            if hint_match:
                return MasterActionMessage(
                    hint_word=hint_match.group(1),
                    hint_number=int(hint_match.group(2))
                )
            return None
        except Exception:
            return None

    def _parse_player_response(self, response: str) -> PlayerActionMessage:
        """Parse the player's response to extract guesses."""
        try:
            json_match = re.search(r'\{[^{}]*"guesses"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return PlayerActionMessage(guesses=data.get("guesses", []))
            
            # Fallback: try parsing the whole response as JSON
            data = json.loads(response)
            return PlayerActionMessage(guesses=data.get("guesses", []))
        except (json.JSONDecodeError, Exception):
            return None

    def step(self) -> dict:
        """
        Run a complete turn: master gives hint, player guesses.
        Returns the combined results.
        """
        success_flag = True
        m_state = self.get_master_state()
        m_prompt = format_master_prompt(m_state)
        m_response = self._query_ollama(m_prompt, self.master_model)
        
        m_action = self._parse_master_response(m_response)
        if m_action is None:
            success_flag = False
            return {"success": False, "error": "Failed to parse master response", "raw_response": m_response}
    
        master_result = self.handle_master_action(m_action)
        if not master_result.get("success"):
            return {"success": False, "phase": "master", "error": master_result}

        p_state = self.get_player_state()
        p_prompt = format_player_prompt(p_state)
        p_response = self._query_ollama(p_prompt, self.player_model)
        
        player_action = self._parse_player_response(p_response)
        if player_action is None:
            success_flag = False
            return {"success": False, "error": "Failed to parse player response", "raw_response": p_response}
        player_result = self.handle_player_action(player_action)


        self.step_count += 1
        log_event = {
            "master_result": master_result,
            "player_result": player_result,
            "environment_state": self.environment.get_game_state(),
            "master_response": m_response,
            "player_response": p_response,
            "master_action": m_action,
            "player_action": player_action,
            "game_over": self.environment.check_win(),
            "success": success_flag
        }
        self.orchestration_log[self.step_count] = log_event
        self.reward_log[self.step_count] = reward_function(log_event)
        return log_event
    
    def run_episode(self, limit: int = 10) -> dict:
        """Run the full game until completion."""
        while not self.environment.check_win() and self.step_count < limit:
            step_result = self.step()
            if not step_result.get("success"):
                return {"success": False, "complete_log": self.orchestration_log, "reward_log": self.reward_log, "total_steps": self.step_count}
        return {"success": True, "complete_log": self.orchestration_log, "reward_log": self.reward_log, "total_steps": self.step_count}