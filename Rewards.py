def handleReward(event):
    env_data = event.get("environment_state", {})
    board = env_data.get("board")

    master_result = event.get("master_result", "")
    player_result = event.get("player_result", "")
    # Master rewards
    master_reward = 0
    # Format must be correct
    if master_result.get("success") is not True:
        master_reward = -10  # Penalty for invalid format
    elif master_result.get("hint_word") in set(board):
        master_reward = -8  # Penalty for using board word as hint
    else:
        master_reward = 5  # Small reward for valid hint

    
    # Player rewards
    player_reward = 0
    if player_result.get("success") is not True:
        player_reward = -10  # Penalty for invalid format
    else:
        env_results = player_result.get("environment_result", {})
        correct_guesses = sum(1 for res in env_results.get("results", []) if res.get("result") == "correct")
        neutral_guesses = sum(1 for res in env_results.get("results", []) if res.get("result") == "neutral")
        opponent_guesses = sum(1 for res in env_results.get("results", []) if res.get("result") == "opponent")

        player_reward += correct_guesses * 5  # Reward for correct guesses
        player_reward -= neutral_guesses * 2   # Penalty for neutral guesses
        player_reward -= opponent_guesses * 5  # Penalty for opponent guesses

    return master_reward, player_reward