import json
import os
import re
import requests
from Rewards import RewardModule
from configs.Configs import OpenAIConfig
from messages.Message import MasterActionMessage, PlayerActionMessage, PlayerDiscussionMessage
from models.BenchmarkAgent import BenchmarkAgent

class OpenAIAgent(BenchmarkAgent):
    def __init__(self, config:OpenAIConfig):

        self.config = config

        self.api_key = os.environ.get('OPEN_API_KEY')
        
        self.model = config.model
        
        print(f"OpenAI Model initialized")
        
    def generate_player_action(self, prompt:str) -> tuple[PlayerActionMessage, str]:
        response = self._query(prompt)
        return (self._parse_player_response(response), response)
    
    def generate_player_discussion(self, prompt:str, identifier:str, history:list[str]) -> tuple[PlayerDiscussionMessage, str]:
        response = self._query(prompt)
        return (self._parse_player_discussion(response), response)
    
    def generate_master(self, prompt:str) -> tuple[MasterActionMessage, str]:
        response = self._query(prompt)
        return (self._parse_master_response(response), response)
        
    def get_config(self):
        return self.config
        
    def _query(self, prompt: str) -> str:
        """Query the OpenAI API with the given prompt and model."""
        payload = {
            "model": self.model,
            "input": prompt,
        }
        headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
        }
        try:
            resp = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=2000)
            resp.raise_for_status()
            
            data = resp.json()
            # Handle dual output format: a list containing 'reasoning' and 'message'
            outputs = data.get("output", [])
            for item in outputs:
                if item.get("type") == "message":
                    content_list = item.get("content", [])
                    for content_item in content_list:
                        if content_item.get("type") == "output_text":
                            return content_item.get("text", "")
            
            if "output" in data and "content" in data["output"]:
                 return data["output"]["content"]["text"]
                 
            return ""
        except requests.exceptions.Timeout:
            return "Error: Timeout"
        except requests.exceptions.RequestException as e:
            return f"Error querying OpenAI: {e}"
    
    def _parse_master_response(self, response: str) -> MasterActionMessage:
        """Reuse Ollama logic or same logic"""
        try:
            # Look for the <RESULT>...</RESULT> block
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response
            
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
        """Reuse existing logic"""
        try:
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response

            json_match = re.search(r'\{[^{}]*"guesses"[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return PlayerActionMessage(guesses=data.get("guesses", []))
            
            data = json.loads(response)
            return PlayerActionMessage(guesses=data.get("guesses", []))
        except (json.JSONDecodeError, Exception):
            return None
        
    def _parse_player_discussion(self, response: str) -> PlayerDiscussionMessage:
        """Reuse existing logic"""
        try:
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response

            json_match = re.search(r'\{[^{}]*"guesses"[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return PlayerDiscussionMessage(guesses=data.get("guesses", []))
            
            thought_match = re.search(r'<THOUGHT>(.*?)</THOUGHT>', response, re.DOTALL | re.IGNORECASE)
            if thought_match:
                result_text = thought_match.group(1)
            else:
                result_text = response
            
            data = json.loads(response)

            return PlayerDiscussionMessage(response=thought_match,guesses=data.get("guesses", []))
        except (json.JSONDecodeError, Exception):
            return None