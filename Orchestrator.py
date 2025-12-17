from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any
import json
import re

import requests

from Environment import Environment
from messages.Message import MasterStateMessage, PlayerStateMessage, MasterActionMessage, PlayerActionMessage
from prompts.agent_prompts import format_master_prompt, format_player_prompt
from Rewards import RewardModule


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

    @abstractmethod
    def reset(self) -> None:
        pass

    def run_full(self) -> dict:
        pass
    
class OllamaOrchestrator(Orchestrator):
    def __init__(self, config):
        # Helper to convert config to dict safely
        if is_dataclass(config):
            self.config_dict = asdict(config)
        else:
            self.config_dict = config if isinstance(config, dict) else {}

        super().__init__(self.config_dict)
        self.config = config

        self.master_model = self.config_dict.get("master_model", "llama3.2b")
        self.player_model = self.config_dict.get("player_model", "llama3.2b")
        self.ollama_url = self.config_dict.get("ollama_url", "http://localhost:11434/api/generate")

        self.orchestration_log = {}
        self.reward_log = {}
        self.step_count = 0
        
        # Initialize Reward Module
        self.reward_module = RewardModule(self.config_dict.get("reward_config"))
        print(f"Ollama Orchestrator initialized with Master Model: {self.master_model}, Player Model: {self.player_model}")

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

    def step(self) -> dict:
        """
        Run a complete turn: master gives hint, player guesses.
        Returns the combined results.
        """
        self.step_count += 1
        
        # Initialize log data
        m_state = self.get_master_state()
        m_prompt = format_master_prompt(m_state)
        m_response = ""
        m_action = None
        master_result = {}
        
        p_prompt = ""
        p_response = ""
        player_action = None
        player_result = {}
        
        success_flag = False
        error_msg = ""

        try:
            m_response = self._query_ollama(m_prompt, self.master_model)
            m_action = self._parse_master_response(m_response)
            
            if m_action is None:
                error_msg = "Failed to parse master response"
                return self._finalize_step(m_prompt, m_response, m_action, master_result, 
                                         p_prompt, p_response, player_action, player_result, 
                                         False, error_msg)

            master_result = self.handle_master_action(m_action)
            if not master_result.get("success"):
                error_msg = f"Master action failed: {master_result}"
                return self._finalize_step(m_prompt, m_response, m_action, master_result, 
                                         p_prompt, p_response, player_action, player_result, 
                                         False, error_msg)

            p_state = self.get_player_state()
            p_prompt = format_player_prompt(p_state)
            p_response = self._query_ollama(p_prompt, self.player_model)
            
            player_action = self._parse_player_response(p_response)
            if player_action is None:
                error_msg = "Failed to parse player response"
                return self._finalize_step(m_prompt, m_response, m_action, master_result, 
                                         p_prompt, p_response, player_action, player_result, 
                                         False, error_msg)

            player_result = self.handle_player_action(player_action)
            success_flag = True
            
        except Exception as e:
            error_msg = str(e)
            success_flag = False
        
        return self._finalize_step(m_prompt, m_response, m_action, master_result, 
                                 p_prompt, p_response, player_action, player_result, 
                                 success_flag, error_msg)
    
    def run_episode(self, limit: int = 10) -> dict:
        """Run the full game until completion."""
        while not self.environment.check_win() and self.step_count < limit:
            step_result = self.step()
            if not step_result.get("success"):
                return {"success": False, "complete_log": self.orchestration_log, "reward_log": self.reward_log, "total_steps": self.step_count}
        return {"success": True, "complete_log": self.orchestration_log, "reward_log": self.reward_log, "total_steps": self.step_count}
    
    def save_run_log(self, filepath: str, run_id: str):
        """Save the orchestration log to a JSON file."""
        run_data = {
            "run_id": run_id,
            "config": self.config_dict,
            "orchestration_log": self.orchestration_log,
            "reward_log": self.reward_log
        }
        with open(filepath, 'w') as f:
            json.dump(run_data, f, indent=4)

    def reset(self) -> None:
        """Reset the orchestrator for a new episode."""
        self.environment = Environment(self.config_dict.get("env_config"))
        self.orchestration_log = {}
        self.reward_log = {}
        self.step_count = 0

    def _finalize_step(self, m_prompt, m_resp, m_action, m_result, 
                      p_prompt, p_resp, p_action, p_result, 
                      success, error_msg=""
                      ) -> dict:
        
        log_event = {
            "step": self.step_count,
            "success": success,
            "error": error_msg,
            "environment_state": self.environment.get_game_state(),
            
            "master_prompt": m_prompt,
            "master_response": m_resp,
            "master_action": m_action.to_dict() if m_action else None,
            "master_result": m_result,
            
            "player_prompt": p_prompt,
            "player_response": p_resp,
            "player_action": p_action.to_dict() if p_action else None,
            "player_result": p_result,
            
            "game_over": self.environment.check_win()
        }
        
        self.orchestration_log[self.step_count] = log_event
        try:
            self.reward_log[self.step_count] = self.reward_module.reward_function(log_event)
        except Exception:
            self.reward_log[self.step_count] = (0, 0)
            
        return log_event

    def _query_ollama(self, prompt: str, model: str) -> str:
        """Query the Ollama API with the given prompt and model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.Timeout:
            return "Error: Timeout"
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
            
            # Robust regex to handle quotes and spacing
            # Matches: HINT: "apple" NUMBER: 2 Or HINT: apple NUMBER: 2
            hint_match = re.search(r'HINT:\s*["\']?([\w-]+)["\']?\s*NUMBER:\s*(\d+)', result_text, re.IGNORECASE)
            if hint_match:
                return MasterActionMessage(
                    hint_word=hint_match.group(1),
                    hint_number=int(hint_match.group(2))
                )
            
            return None
        except Exception as e:
            print(f"Exception during Master parsing: {e}")
            return None

    def _parse_player_response(self, response: str) -> PlayerActionMessage:
        """Parse the player's response to extract guesses."""
        try:
            # First, try to find the <RESULT> block
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response

            json_match = re.search(r'\{[^{}]*"guesses"[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return PlayerActionMessage(guesses=data.get("guesses", []))
            
            # Fallback: try parsing the whole response as JSON
            data = json.loads(response)
            return PlayerActionMessage(guesses=data.get("guesses", []))
        except (json.JSONDecodeError, Exception):
            return None
    
class OpenAIOrchestrator(Orchestrator):
    def __init__(self, config, api_key):
        # Helper to convert config to dict safely
        if is_dataclass(config):
            self.config_dict = asdict(config)
        else:
            self.config_dict = config if isinstance(config, dict) else {}

        super().__init__(self.config_dict)
        self.config = config

        self.master_model = self.config_dict.get("master_model", "gpt-5-nano")
        self.player_model = self.config_dict.get("player_model", "gpt-5-nano")
        self.api_key = api_key

        self.orchestration_log = {}
        self.reward_log = {}
        self.step_count = 0
        
        # Initialize Reward Module
        self.reward_module = RewardModule(self.config_dict.get("reward_config"))
        print(f"OpenAI Orchestrator initialized with Master Model: {self.master_model}, Player Model: {self.player_model}")

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

    def handle_master_action(self, action: MasterActionMessage) -> dict:
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

    def handle_player_action(self, action: PlayerActionMessage) -> dict:
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
    
    def step(self) -> dict:
        """
        Run a complete turn: master gives hint, player guesses.
        Returns the combined results.
        """
        self.step_count += 1
        
        # Initialize log data
        m_state = self.get_master_state()
        m_prompt = format_master_prompt(m_state)
        m_response = ""
        m_action = None
        master_result = {}
        
        p_prompt = ""
        p_response = ""
        player_action = None
        player_result = {}
        
        success_flag = False
        error_msg = ""

        try:
            m_response = self._query_openai(m_prompt, self.master_model)
            m_action = self._parse_master_response(m_response)
            
            if m_action is None:
                error_msg = "Failed to parse master response"
                return self._finalize_step(m_prompt, m_response, m_action, master_result, 
                                         p_prompt, p_response, player_action, player_result, 
                                         False, error_msg)

            master_result = self.handle_master_action(m_action)
            if not master_result.get("success"):
                error_msg = f"Master action failed: {master_result}"
                return self._finalize_step(m_prompt, m_response, m_action, master_result, 
                                         p_prompt, p_response, player_action, player_result, 
                                         False, error_msg)

            p_state = self.get_player_state()
            p_prompt = format_player_prompt(p_state)
            p_response = self._query_openai(p_prompt, self.player_model)
            
            player_action = self._parse_player_response(p_response)
            if player_action is None:
                error_msg = "Failed to parse player response"
                return self._finalize_step(m_prompt, m_response, m_action, master_result, 
                                         p_prompt, p_response, player_action, player_result, 
                                         False, error_msg)

            player_result = self.handle_player_action(player_action)
            success_flag = True
            
        except Exception as e:
            error_msg = str(e)
            success_flag = False
        
        return self._finalize_step(m_prompt, m_response, m_action, master_result, 
                                 p_prompt, p_response, player_action, player_result, 
                                 success_flag, error_msg)
        
    def run_episode(self, limit: int = 10) -> dict:
        """Run the full game until completion."""
        while not self.environment.check_win() and self.step_count < limit:
            step_result = self.step()
            if not step_result.get("success"):
                return {"success": False, "complete_log": self.orchestration_log, "reward_log": self.reward_log, "total_steps": self.step_count}
        return {"success": True, "complete_log": self.orchestration_log, "reward_log": self.reward_log, "total_steps": self.step_count}
    
    def save_run_log(self, filepath: str, run_id: str):
        """Save the orchestration log to a JSON file."""
        
        run_data = {
            "run_id": run_id,
            "config": self.config_dict,
            "orchestration_log": self.orchestration_log,
            "reward_log": self.reward_log
        }
        with open(filepath, 'w') as f:
            json.dump(run_data, f, indent=4)

    def reset(self) -> None:
        """Reset the orchestrator for a new episode."""
        self.environment = Environment(self.config_dict.get("env_config"))
        self.orchestration_log = {}
        self.reward_log = {}
        self.step_count = 0

    def _finalize_step(self, m_prompt, m_resp, m_action, m_result, 
                      p_prompt, p_resp, p_action, p_result, 
                      success, error_msg=""
                      ) -> dict:
        
        log_event = {
            "step": self.step_count,
            "success": success,
            "error": error_msg,
            "environment_state": self.environment.get_game_state(),
            
            "master_prompt": m_prompt,
            "master_response": m_resp,
            "master_action": m_action.to_dict() if m_action else None,
            "master_result": m_result,
            
            "player_prompt": p_prompt,
            "player_response": p_resp,
            "player_action": p_action.to_dict() if p_action else None,
            "player_result": p_result,
            
            "game_over": self.environment.check_win()
        }
        
        self.orchestration_log[self.step_count] = log_event
        try:
            self.reward_log[self.step_count] = self.reward_module.reward_function(log_event)
        except Exception:
            self.reward_log[self.step_count] = (0, 0)
            
        return log_event

    def _query_openai(self, prompt: str, model: str) -> str:
        """Query the OpenAI API with the given prompt and model."""
        payload = {
            "model": model,
            "input": prompt,
        }
        headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
        }
        try:
            resp = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=2000)
            resp.raise_for_status()
            return resp.json()["output"]["content"]["text"]
        except requests.exceptions.Timeout:
            return "Error: Timeout"
        except requests.exceptions.RequestException as e:
            return f"Error querying OpenAI: {e}"
    
    def _parse_master_response(self, response: str) -> MasterActionMessage:
        """Parse the master's response to extract hint word and number."""
        try:
            # Look for the <RESULT>...</RESULT> block
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response
            
            # Robust regex to handle quotes and spacing
            # Matches: HINT: "apple" NUMBER: 2 Or HINT: apple NUMBER: 2
            hint_match = re.search(r'HINT:\s*["\']?([\w-]+)["\']?\s*NUMBER:\s*(\d+)', result_text, re.IGNORECASE)
            if hint_match:
                return MasterActionMessage(
                    hint_word=hint_match.group(1),
                    hint_number=int(hint_match.group(2))
                )
            
            return None
        except Exception as e:
            print(f"Exception during Master parsing: {e}")
            return None

    def _parse_player_response(self, response: str) -> PlayerActionMessage:
        """Parse the player's response to extract guesses."""
        try:
            # First, try to find the <RESULT> block
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response

            json_match = re.search(r'\{[^{}]*"guesses"[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return PlayerActionMessage(guesses=data.get("guesses", []))
            
            # Fallback: try parsing the whole response as JSON
            data = json.loads(response)
            return PlayerActionMessage(guesses=data.get("guesses", []))
        except (json.JSONDecodeError, Exception):
            return None

    def run_full(self) -> dict:
        pass