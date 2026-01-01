import copy
from Orchestrator import Orchestrator

from configs.Configs import OpenAIConfig, EnvironmentConfig, OrchestratorConfig, RewardConfig, TeamConfig

from dotenv import load_dotenv

import uuid
import os

from models.OpenAIAgent import OpenAIAgent

load_dotenv()

api_key = os.environ.get('OPEN_API_KEY')

OPENAI_CONFIG = OpenAIConfig(
    model="gpt-5-nano"
)

MODEL = OpenAIAgent(
    config=OPENAI_CONFIG
)

TEAM_CONFIG = TeamConfig(
    master_model=MODEL,
    player_models=[copy.copy(MODEL) for _ in range(2)]  # Two players per team
)

ORCHESTRATOR_CONFIG = OrchestratorConfig(
    team_configs=[TEAM_CONFIG],
    env_config=EnvironmentConfig(
        teams=1,
        max_words=25,
        word_list_file="./content/wordlist.txt"
    ),
    reward_config=RewardConfig(),
)

orchestrator = Orchestrator(ORCHESTRATOR_CONFIG)
result = orchestrator.run_episode(limit=10)
run_id = str(uuid.uuid4())
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
orchestrator.save_run_log(f"{output_dir}/openai_run_{run_id}.json", run_id)
print(f"Run complete. Log saved to {output_dir}/openai_run_{run_id}.json")
