import json
import re
import requests
from configs.Configs import OllamaConfig
from messages.Message import MasterActionMessage, PlayerActionMessage, PlayerDiscussionMessage
from models.BenchmarkAgent import BenchmarkAgent

class OllamaModel(BenchmarkAgent):
    
    def __init__(self, config:OllamaConfig):
        self.config = config
        
        self.model = config.model
        
        self.ollama_url = config.ollama_url
        
        print(f"OllamaModel Model initialized")
  
    def generate_player_action(self, prompt:str) -> tuple[PlayerActionMessage, str]:
        response = self._query(prompt)
        return (self._parse_player_response(response), response)
    
    def generate_player_discussion(self, prompt:str, identifier:str, history:str) -> tuple[PlayerDiscussionMessage, str]:
        response = self._query(prompt)
        return (self._parse_player_discussion(response), response)
    
    def generate_master(self, prompt:str) -> tuple[MasterActionMessage, str]:
        response = self._query(prompt)
        return (self._parse_master_response(response), response)
    
    def get_config(self):
        return self.config
    
    def _query(self, prompt: str) -> str:
        """Query the Ollama API with the given prompt and model."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.Timeout:
            return "Error: Timeout"
        except requests.exceptions.RequestException as e:
            return f"Error querying Ollama: {e}"

    def _parse_master_response(self, response: str) -> MasterActionMessage:
        """Parse the master's response to extract hint word and number."""
        try:
            # Look for the <RESULT>...</RESULT> block
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response
            
            # Robust regex to handle quotes and spacing
            # Matches: HINT: "apple" NUMBER: 2 Or HINT: apple NUMBER: 2
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
        """Parse the player's response to extract guesses."""
        try:
            # First, try to find the <RESULT> block
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_text = result_match.group(1)
            else:
                result_text = response

            json_match = re.search(r'\{[^{}]*"guesses"[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return PlayerActionMessage(guesses=data.get("guesses", []))
            
            # Fallback: try parsing the whole response as JSON
            data = json.loads(response)
            return PlayerActionMessage(guesses=data.get("guesses", []))
        except (json.JSONDecodeError, Exception):
            return None