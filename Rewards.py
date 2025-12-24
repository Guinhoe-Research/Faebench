class RewardModule:
    def __init__(self, reward_config=None):
        if isinstance(reward_config, dict):
            self.FORMAT_PENALTY = reward_config.get("FORMAT_PENALTY", -10) if reward_config else -10
        else:
            self.FORMAT_PENALTY = reward_config.FORMAT_PENALTY if reward_config else -10
        self.BOARD_WORD_USE_PENALTY = -8
        self.VALID_HINT_REWARD = 5

        self.CORRECT_GUESS_REWARD_WEIGHT = 5
        self.NEUTRAL_GUESS_PENALTY_WEIGHT = 2
        self.OPPONENT_GUESS_PENALTY_WEIGHT = 5

    def reward_function(self, event):
        env_data = event.get("environment_state", {})
        board = env_data.get("board")

        master_result = event.get("master_result", "")
        player_result = event.get("player_result", "")
        # Master rewards
        master_reward = 0
        # Format must be correct
        if master_result.get("success") is not True:
            master_reward = self.FORMAT_PENALTY  # Penalty for invalid format
        elif master_result.get("action", {}).get("hint_word") in set(board):
            master_reward = self.BOARD_WORD_USE_PENALTY  # Penalty for using board word as hint
        else:
            master_reward = self.VALID_HINT_REWARD  # Small reward for valid hint
        
        # Player rewards
        player_reward = 0
        if player_result.get("success") is not True:
            player_reward = self.FORMAT_PENALTY  # Penalty for invalid format
        else:
            env_results = player_result.get("result", {})

            for res in env_results.get("results", []):
                word, result = res.get("word"), res.get("result")
                
                if result == "already_guessed":
                    player_reward -= 12
                elif result == "correct":
                    player_reward += self.CORRECT_GUESS_REWARD_WEIGHT
                elif result == "neutral":
                    player_reward -= self.NEUTRAL_GUESS_PENALTY_WEIGHT
                elif result == "opponent":
                    player_reward -= self.OPPONENT_GUESS_PENALTY_WEIGHT
        return master_reward, player_reward