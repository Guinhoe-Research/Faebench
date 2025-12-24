from dataclasses import asdict, dataclass, is_dataclass
from models.BenchmarkAgent import BenchmarkAgent

@dataclass
class OrchestratorConfig:
    team_configs: list['TeamConfig']
    env_config: 'EnvironmentConfig'
    reward_config: 'RewardConfig'

    def to_dict(self):
        env_config = self.env_config
        reward_config = self.reward_config
        return {
            "team_configs": [
                m.to_dict() if hasattr(m, "to_dict") else m
                for m in self.team_configs
            ],
            "env_config": asdict(env_config) if is_dataclass(env_config) else env_config,
            "reward_config": asdict(reward_config) if is_dataclass(reward_config) else reward_config,
        }
@dataclass
class OpenAIConfig:
    model: str
    
    def to_dict(self):
        return {
            "model": self.model
        }

@dataclass
class OllamaConfig:
    model: str
    ollama_url: str

    def to_dict(self):
        return {
            "model": self.model,
            "ollama_url": self.ollama_url
        }

@dataclass
class RewardConfig:
    # Penalty for incorrect formatting of both Master and Player actions
    FORMAT_PENALTY: int = -10
    # Penalty for using a board word as a hint
    BOARD_WORD_USE_PENALTY: int = -8
    # Reward for providing a valid hint (For Master)
    VALID_HINT_REWARD: int = 5
    # Reward for a correct guess (For Player)
    CORRECT_GUESS_REWARD_WEIGHT: int = 5
    # Penalty for a neutral guess (For Player)
    NEUTRAL_GUESS_PENALTY_WEIGHT: int = 2
    # Penalty for an opponent guess (For Player)
    OPPONENT_GUESS_PENALTY_WEIGHT: int = 5

@dataclass
class EnvironmentConfig:
    teams: int
    max_words: int
    word_list_file: str

@dataclass
class TeamConfig:
    master_model: 'BenchmarkAgent'
    player_models: list['BenchmarkAgent']  
    
    def to_dict(self):
        master_config = self.master_model.get_config()
        return {
            "master_model": master_config.to_dict() if hasattr(master_config, "to_dict") else master_config,
            "player_models": [
                m.get_config().to_dict() if hasattr(m.get_config(), "to_dict") else m.get_config()
                for m in self.player_models
            ]
        }