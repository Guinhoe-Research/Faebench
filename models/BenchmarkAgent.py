from abc import ABC, abstractmethod
from messages.Message import MasterActionMessage, PlayerActionMessage, PlayerDiscussionMessage

class BenchmarkAgent(ABC):
  
    def __init__(self):
      pass
  
    @abstractmethod
    def generate_player_action(self, prompt: str) -> tuple[PlayerActionMessage, str]:
      pass
    
    @abstractmethod
    def generate_player_discussion(self, prompt:str, identifier:str, history:list[str]) -> tuple[PlayerDiscussionMessage, str]:
      pass
    
    @abstractmethod
    def generate_master(self, prompt: str) -> tuple[MasterActionMessage, str]:
      pass
    
    @abstractmethod
    def get_config(self):
      pass