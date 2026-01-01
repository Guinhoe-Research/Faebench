
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Adjust path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Orchestrator import Orchestrator
from Environment import Environment as EnvironmentStandard
from messages.Message import MasterActionMessage, PlayerActionMessage

class TestOrchestratorStandard(unittest.TestCase):
    def setUp(self):
        # 1. Setup Environment with User Configuration
        config = {"word_list_file": "content/wordlist.txt", "teams": 1, "test_flag": True} 
        self.env = EnvironmentStandard(config)
        
        # Manual override of board state
        self.env.board = ["wheat", "water", "lava", "netherrack", "wood", "door", "iron", "gold", "diamond", "emerald"]
        self.env.word_sets = {
            1: ["gold", "diamond", "emerald"]
        }
        self.env.neutral_words = ["netherrack", "wood", "door"]
        
        # 2. Setup Orchestrator
        self.orch = Orchestrator({"env_config": config, "team_configs": [{"master_model": "mock", "player_models": ["mock"]}]})
        self.orch.environment = self.env
        
        # Mock Models
        self.mock_master = MagicMock()
        self.mock_player = MagicMock()
        self.orch.teams[1]["master_model"] = self.mock_master
        self.orch.teams[1]["player_models"] = [self.mock_player]

    def test_full_step_execution(self):
        print("\n--- Testing Standard Orchestrator Step ---")
        
        # Mock Responses
        self.mock_master.generate_master.return_value = (
            MasterActionMessage(hint_word="precious", hint_number=2),
            "HINT: 'precious' NUMBER: 2"
        )
        
        self.mock_player.generate_player_action.return_value = (
            PlayerActionMessage(guesses=["gold", "diamond"]),
            '{"guesses": ["gold", "diamond"]}'
        )

        step_result = self.orch.step()
        
        log_entry = self.orch.orchestration_log[1]

        # Assert specific logs
        master_prompt = log_entry['team_logs'][1]['master_prompt']
        self.assertIn("gold", str(master_prompt))
        self.assertIn("diamond", str(master_prompt))
        
        self.assertEqual(log_entry['team_logs'][1]['master_action']['hint_word'], "precious")
        self.assertEqual(log_entry['team_logs'][1]['master_action']['hint_number'], 2)
        
        player_prompt = log_entry['team_logs'][1]['player_prompt']
        self.assertIn("precious", str(player_prompt))
        
        result = step_result['team_logs'][1]['player_result']
        self.assertEqual(result['result']['correct_count'], 2)
        self.assertEqual(step_result['team_logs'][1]['success'], True)
        
        self.assertIn("gold", self.env.guessed_words)
        self.assertIn("diamond", self.env.guessed_words)


class TestOrchestratorMultiteam(unittest.TestCase):
    def setUp(self):
        config = {"teams": 2, "max_words": 10, "test_flag": True}
        self.env = EnvironmentStandard(config)
        
        self.env.board = ["wheat", "water", "lava", "netherrack", "wood", "door", "iron", "gold", "diamond", "emerald"]
        self.env.word_sets = {
            1: ["gold", "diamond", "emerald"],
            2: ["wheat", "water", "lava"]
        }
        self.env.neutral_words = ["netherrack", "wood", "door"]

        self.orch = Orchestrator({
            "env_config": config, 
            "team_configs": [
                {"master_model": "mock", "player_models": ["mock"]}, 
                {"master_model": "mock", "player_models": ["mock"]}
            ]
        })
        self.orch.environment = self.env
        
        self.orch.teams[1]["master_model"] = MagicMock()
        self.orch.teams[1]["player_models"] = [MagicMock()]
        
        self.orch.teams[2]["master_model"] = MagicMock()
        self.orch.teams[2]["player_models"] = [MagicMock()]

    def test_team1_step(self):
        print("\n--- Testing Multiteam Orchestrator Team 1 ---")
        
        mock_master = self.orch.teams[1]["master_model"]
        mock_player = self.orch.teams[1]["player_models"][0]
        
        mock_master.generate_master.return_value = (
            MasterActionMessage(hint_word="shiny", hint_number=1),
            "HINT: shiny NUMBER: 1"
        )
        mock_player.generate_player_action.return_value = (
            PlayerActionMessage(guesses=["gold"]),
            '{"guesses": ["gold"]}'
        )
        
        step_result = self.orch.team_step(1)
        
        self.assertEqual(step_result['master_action']['hint_word'], "shiny")
        self.assertEqual(step_result['player_result']['result']['correct_count'], 1)
        self.assertIn("gold", self.env.guessed_words)

    def test_team2_step_fail(self):
        print("\n--- Testing Multiteam Orchestrator Team 2 (Hit Opponent) ---")
        
        mock_master = self.orch.teams[2]["master_model"]
        mock_player = self.orch.teams[2]["player_models"][0]
        
        mock_master.generate_master.return_value = (
            MasterActionMessage(hint_word="liquid", hint_number=2),
            "HINT: liquid NUMBER: 2"
        )
        mock_player.generate_player_action.return_value = (
            PlayerActionMessage(guesses=["water", "gold"]),
            '{"guesses": ["water", "gold"]}'
        )
        
        step_result = self.orch.team_step(2)
        
        results = step_result['player_result']['result']['results']
        self.assertEqual(results[0]['word'], "water")
        self.assertEqual(results[1]['word'], "gold")
        
        self.assertIn("water", self.env.guessed_words)
        self.assertIn("gold", self.env.guessed_words)

    def test_multi_player_consensus(self):
        """
        Team 1 has 2 player models.
        - Master: "shiny 1"
        - Player 1: "gold" -> "gold"
        - Player 2: "diamond" -> "gold"
        - Judge: "gold"
        """
        print("\n--- Testing Multi-Player Consensus ---")
        
        mock_p1 = MagicMock()
        mock_p2 = MagicMock()
        
        # P1 responses
        mock_p1.generate_player_action.side_effect = [
            (PlayerActionMessage(guesses=["gold"]), "Thought: gold is shiny"), # Round 1
            (PlayerActionMessage(guesses=["gold"]), "Refined: gold still shiny") # Round 2
        ]
        
        # P2 responses
        mock_p2.generate_player_action.side_effect = [
            (PlayerActionMessage(guesses=["diamond"]), "Thought: diamond is shiny"), # Round 1
            (PlayerActionMessage(guesses=["gold"]), "Refined: gold is better"), # Round 2
            (PlayerActionMessage(guesses=["gold"]), "Judge: Consensus is gold") # Judge
        ]
        
        self.orch.teams[1]["player_models"] = [mock_p1, mock_p2]
        mock_master = self.orch.teams[1]["master_model"]
        
        # Note: If reusing mock_master from setUp, reset it or overwrite return value
        mock_master.generate_master.return_value = (
            MasterActionMessage(hint_word="shiny", hint_number=1),
            "Hint: shiny 1"
        )
        
        step_result = self.orch.team_step(1)
        
        # Assert SUCCESS first
        self.assertEqual(step_result['error'], "", f"Step failed with error: {step_result['error']}")
        
        # Verify Results
        p_result = step_result['player_result']
        self.assertEqual(p_result['result']['correct_count'], 1)
        self.assertEqual(p_result['result']['results'][0]['word'], "gold")
        
        # Verify Model Calls
        self.assertEqual(mock_p1.generate_player_action.call_count, 2)
        self.assertEqual(mock_p2.generate_player_action.call_count, 3)

if __name__ == "__main__":
    unittest.main()
