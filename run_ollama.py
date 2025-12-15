from Orchestrator import OllamaOrchestrator

from configs.Configs import OrchestratorConfig, EnvironmentConfig, RewardConfig
import uuid
import os


CONFIG = OrchestratorConfig(
    master_model="llama3.2",
    player_model="llama3.2",
    env_config=EnvironmentConfig(
        teams=1,
        max_words=25,
        word_list_file="Zigong/content/wordlist.txt"
    ),
    reward_config=RewardConfig()
)

orchestrator = OllamaOrchestrator(CONFIG)
result = orchestrator.run_episode(limit=10)
run_id = str(uuid.uuid4())
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
orchestrator.save_run_log(f"{output_dir}/ollama_run_{run_id}.json", run_id)
print(f"Run complete. Log saved to {output_dir}/ollama_run_{run_id}.json")
