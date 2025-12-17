from dataclasses import dataclass

@dataclass
class OrchestratorConfig:
    master_model: str
    player_model: str
    env_config: 'EnvironmentConfig'
    reward_config: 'RewardConfig'
    
@dataclass
class OpenAIConfig:
    master_model: str
    player_model: str
    env_config: 'EnvironmentConfig'
    reward_config: 'RewardConfig'

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