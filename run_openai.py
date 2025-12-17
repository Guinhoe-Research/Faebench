from Orchestrator import OpenAIOrchestrator

from configs.Configs import OpenAIConfig, EnvironmentConfig, RewardConfig

from dotenv import load_dotenv

import uuid
import os

load_dotenv()

api_key = os.environ.get('OPEN_API_KEY')

CONFIG = OpenAIConfig(
    master_model="gpt-5-nano",
    player_model="gpt-5-nano",
    env_config=EnvironmentConfig(
        teams=1,
        max_words=25,
        word_list_file="./content/wordlist.txt"
    ),
    reward_config=RewardConfig(),
)

orchestrator = OpenAIOrchestrator(CONFIG, api_key)
result = orchestrator.run_episode(limit=10)
run_id = str(uuid.uuid4())
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
orchestrator.save_run_log(f"{output_dir}/openai_run_{run_id}.json", run_id)
print(f"Run complete. Log saved to {output_dir}/openai_run_{run_id}.json")
