from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
import json
import re

import requests

from Environment import Environment
from configs.Configs import OrchestratorConfig
from messages.Message import MasterStateMessage, PlayerStateMessage, MasterActionMessage, PlayerActionMessage
from prompts.agent_prompts import format_master_prompt, format_player_prompt
from Rewards import RewardModule

class Orchestrator:
    def __init__(self, orchestration_config):
        # Support both dictionary and object configuration
        if isinstance(orchestration_config, dict):
            self.environment = Environment(orchestration_config.get("env_config"))
            self.teams = {i+1: {
                "master_model": team_cfg.get("master_model", "llama3.2b"),
                "player_models": team_cfg.get("player_models", ["llama3.2b"])
            } for i, team_cfg in enumerate(orchestration_config.get("team_configs", []))}
            self.reward_module = RewardModule(orchestration_config.get("reward_config"))
        else:
            self.environment = Environment(orchestration_config.env_config)
            self.teams = {i+1: {
                "master_model": team_cfg.master_model,
                "player_models": team_cfg.player_models
            } for i, team_cfg in enumerate(orchestration_config.team_configs)}
            self.reward_module = RewardModule(orchestration_config.reward_config)
        
        self.orchestration_log = {}
        self.reward_log = {i: [] for i in self.teams.keys()}
        self.step_count = 0
        
        if isinstance(orchestration_config, OrchestratorConfig):
            self.config_dict = orchestration_config.to_dict()
        else:
            self.config_dict = orchestration_config if isinstance(orchestration_config, dict) else {}

    def get_master_state(self, team_id=1) -> MasterStateMessage:
        """Get the current game state formatted for the codemaster."""
        env_state = self.environment.get_master_state(team_id)
        
        # Get team words LEFT for the specified master
        # Note: Environment now ensures 'word_sets' is returned as a dict in get_master_state
        team_words = [    
            i for i in env_state["word_sets"].get(team_id, [])
            if i not in env_state["guessed_words"]
        ]
        
        # Determine opponent words (all other teams' words)
        opponent_words = []
        for tid, words in env_state["word_sets"].items():
            if team_id != tid:
                opponent_words.extend(words)
        
        # Neutral words are board words not assigned to any team
        all_team_words = set()
        for words in env_state["word_sets"].values():
            all_team_words.update(words)
        neutral_words = [w for w in env_state["board"] if w not in all_team_words]
        
        guessed_words_log = env_state["guessed_words_log"]
        
        return MasterStateMessage(
            team_words=team_words,
            opponent_words=opponent_words,
            neutral_words=neutral_words,
            guessed_words_log=guessed_words_log
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
            "result": result
        }
    
    def get_player_state(self, hint: dict) -> PlayerStateMessage:
        """Get the current game state formatted for a player."""
        env_state = self.environment.get_player_state(team_id=1) # Defaulting to 1 for generic call, but see team_step override
        
        return PlayerStateMessage(
            hint_word=hint.get("word", ""),
            hint_number=hint.get("number", 0),
            board=env_state.get("board", []),
            guessed_words_log=env_state.get("guessed_words_log", [])
        )
        
    def get_player_state_for_team(self, hint: dict, team_id: int) -> PlayerStateMessage:
        env_state = self.environment.get_player_state(team_id)
        return PlayerStateMessage(
            hint_word=hint.get("word", ""),
            hint_number=hint.get("number", 0),
            board=env_state.get("board", []),
            guessed_words_log=env_state.get("guessed_words_log", [])
        )

    def handle_player_action(self, action: PlayerActionMessage, team_id: int = 1) -> dict:
        """
        Query the player model for guesses and pass them to the environment.
        """
        result = self.environment.handle_player_action(action, team_id)
        
        return {
            "success": True,
            "action": action.to_dict(),
            "result": result
        }
    
    def team_step(self, team_id: int) -> dict:
        error_msg = ""

        m_state: MasterStateMessage = self.get_master_state(team_id)
        m_prompt: str = format_master_prompt(m_state)
        m_response: str = ""
        m_action: MasterActionMessage = None
        m_result: dict = {}

        p_prompt: str = ""
        p_response: str = ""
        p_action: PlayerActionMessage = None
        p_result: dict = {}

        success_flag = False
        try:
            # Query Master
            m_action, m_response = self.teams[team_id]["master_model"].generate_master(m_prompt)
            print(f"[DEBUG] Team {team_id} Master Response:", m_response)
            print(f"[DEBUG] Team {team_id} Master Action:", m_action)
            if m_action is None:
                error_msg = "Failed to parse master response"
            
            m_result = self.handle_master_action(m_action)
            print(f"[DEBUG] Team {team_id} Master Result:", m_result)
            if not m_result.get("success"):
                error_msg = f"Master action failed: {m_result}"
            
            # Query Player
            # Use specialized get_player_state_for_team to ensure correct log retrieval
            p_state = self.get_player_state_for_team(m_result["result"]["hint"], team_id)
            print(f"[DEBUG] Team {team_id} Player State:", p_state)
            p_prompt = format_player_prompt(p_state)

            if len(self.teams[team_id]["player_models"]) == 1:
                p_action, p_response = self.teams[team_id]["player_models"][0].generate_player_action(p_prompt)
            else:
                # TODO: Implement multi-player querying logic
                print(f"[WARNING] Multiple player models found for team {team_id}, but logic not implemented.")
                pass

            # p_action = self._parse_player_response(p_response)
            print(f"[DEBUG] Team {team_id} Player Action:", p_action)
            if p_action is None:
                error_msg = "Failed to parse player response"
            
            p_result = self.handle_player_action(p_action, team_id)
            print(f"[DEBUG] Team {team_id} Player Result:", p_result)
        except Exception as e:
            error_msg = str(e)
        
        success_flag = error_msg == ""
        return {
            "success": success_flag,
            "error": error_msg,
            "environment_state": self.environment.get_game_state(),
            
            "master_prompt": m_prompt,
            "master_response": m_response,
            "master_action": m_action.to_dict() if m_action else None,
            "master_result": m_result,
            
            "player_prompt": p_prompt,
            "player_response": p_response,
            "player_action": p_action.to_dict() if p_action else None,
            "player_result": p_result,
        }
                                                               
    def step(self) -> dict:
        """
        Run a complete turn: master gives hint, player guesses.
        Returns the combined results.
        """
        self.step_count += 1
        overall_log = {}
        for team_id in self.teams.keys():
            print("[LOG] Team", team_id, "Step", self.step_count)
            team_log = self.team_step(team_id)
            overall_log[team_id] = team_log
        
        return self._finalize_step(overall_log)
    
    def run_episode(self, limit: int = 10) -> dict:
        """Run the full game until completion."""
        while not self.environment.check_win() and self.step_count < limit:
            step_result = self.step()
            
        winner = self.environment.get_winner()
        return {
            "success": True, 
            "winner": winner,
            "complete_log": self.orchestration_log, 
            "reward_log": self.reward_log, 
            "total_steps": self.step_count
        }
    
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
        self.reward_log = {i: [] for i in self.teams.keys()}
        self.step_count = 0

    def _finalize_step(self, team_logs) -> dict:
        
        log_event = {
            "step": self.step_count,
            "environment_state": self.environment.get_game_state(),
            "game_over": self.environment.check_win(),
            "team_logs": team_logs
        }
        
        self.orchestration_log[self.step_count] = log_event
        for i in team_logs.keys():
            master_reward, player_reward = 0, 0
            try:
                master_reward, player_reward = self.reward_module.reward_function(team_logs[i])
            except Exception:
                pass
            
            # Ensure reward_log key exists
            if i not in self.reward_log:
                self.reward_log[i] = []
            self.reward_log[i].append((master_reward, player_reward))
                
        return log_event
