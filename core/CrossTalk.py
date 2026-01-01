from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import sys
import os
# Add parent directory to path to allow imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.reasoning_prompts import format_cross_pollination_prompt, format_judge_prompt
from models.BenchmarkAgent import BenchmarkAgent
import json

# Helper for reducer
def merge_dicts(a: Dict, b: Dict) -> Dict:
    return {**a, **b}

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    # responses: key is model_idx, value is { "thought": str, "action": obj, "raw": str }
    round_1_responses: Annotated[Dict[int, Any], merge_dicts]
    round_2_responses: Annotated[Dict[int, Any], merge_dicts]
    final_action: Any
    
    # Context data
    initial_prompt: str
    hint_text: str

def create_independent_node(model_idx, model: BenchmarkAgent):
    """
    Node for Round 1: Independent Generation
    """
    def node(state: AgentState) -> dict:
        prompt = state["initial_prompt"]
        # Agents are stateless executors
        action, raw_response = model.generate_player_action(prompt)
        
        # TODO: We need to extract the <THOUGHT> block if possible, or just use the full raw response as "thought"
        # Since the Orchestrator expects parsed actions, we'll store everything.
        return {
            "round_1_responses": {
                model_idx: {
                    "action": action,
                    "raw": raw_response
                }
            }
        }
    return node

def create_cross_pollination_node(model_idx, model, total_models):
    """
    Node for Round 2: Cross Pollination
    """
    def node(state: AgentState) -> dict:
        # 1. Gather other thoughts from Round 1
        my_prev_response = state["round_1_responses"][model_idx]["raw"]
        
        others_thoughts = []
        for i in range(total_models):
            if i == model_idx: 
                continue
            others_thoughts.append(f"Teammate {i+1}:\n{state['round_1_responses'][i]['raw']}")
            
        teammate_thoughts_text = "\n\n".join(others_thoughts)
        
        # 2. Format prompt
        # We need to parse previous thought from raw response nicely, but for now using raw is fine.
        # Ideally we'd extract content between <THOUGHT> tags.
        
        prompt = format_cross_pollination_prompt(
            hint=state["hint_text"],
            previous_thought=my_prev_response,
            teammate_thoughts=teammate_thoughts_text
        )
        
        # 3. Generate
        action, raw_response = model.generate_player_action(prompt)
        
        return {
            "round_2_responses": {
                model_idx: {
                    "action": action,
                    "raw": raw_response
                }
            }
        }
    return node

def create_judge_node(judge_model, total_models):
    """
    Node for Aggregation: Judge selects best answer
    """
    def node(state: AgentState) -> dict:
        # 1. Gather all Round 2 proposals
        proposals = []
        for i in range(total_models):
            resp = state["round_2_responses"][i]
            proposals.append(f"Proposal {i+1}:\n{resp['raw']}")
            
        team_proposals_text = "\n\n---\n\n".join(proposals)
        
        # 2. Format prompt
        prompt = format_judge_prompt(
            hint=state["hint_text"],
            team_proposals=team_proposals_text
        )
        
        # 3. Generate
        action, raw_response = judge_model.generate_player_action(prompt)
        
        return {
            "final_action": action,
            # We can also add a message to history if we want
            "messages": [raw_response] 
        }
    return node

class CrossTalkModule:
    def __init__(self, player_models: List[BenchmarkAgent]):
        self.player_models = player_models
        self.num_models = len(player_models)
        self.workflow = StateGraph(AgentState)
        
        # Add nodes for Round 1
        for i, model in enumerate(player_models):
            node_name = f"model_{i}_r1"
            self.workflow.add_node(node_name, create_independent_node(i, model))
            self.workflow.add_edge(START, node_name)
            
        # Add nodes for Round 2
        for i, model in enumerate(player_models):
            node_name_r2 = f"model_{i}_r2"
            self.workflow.add_node(node_name_r2, create_cross_pollination_node(i, model, self.num_models))
            
        self.workflow.add_node("sync_1", lambda x: x)
        
        for i in range(self.num_models):
            self.workflow.add_edge(f"model_{i}_r1", "sync_1")
            
        # From sync_1 to all R2
        for i in range(self.num_models):
            self.workflow.add_edge("sync_1", f"model_{i}_r2")
            
        # Add Judge Node
        # Judge is the LAST model in the list (default)
        judge_model = player_models[-1]
        self.workflow.add_node("judge", create_judge_node(judge_model, self.num_models))
        
        # Connect all R2 to Judge
        for i in range(self.num_models):
            self.workflow.add_edge(f"model_{i}_r2", "judge")
            
        self.workflow.add_edge("judge", END)
        
        self.app = self.workflow.compile()

    def execute(self, initial_prompt: str, hint_text: str):
        final_state = self.app.invoke(
            {
                "round_1_responses": {},
                "round_2_responses": {},
                "initial_prompt": initial_prompt,
                "hint_text": hint_text,
                "messages": []
            },
            config={"configurable": {"thread_id": "1"}}
        )

        # Return the action object (PlayerActionMessage) and the raw judge text (as p_response)
        judge_message = final_state["messages"][-1] if final_state["messages"] else ""
        
        # Ensure judge_text is a string, not a Message object
        if hasattr(judge_message, 'content'):
            judge_text = judge_message.content
        else:
            judge_text = str(judge_message)
            
        # Construct full log
        full_log = {
            "round_1": {k: v["raw"] for k, v in final_state["round_1_responses"].items()},
            "round_2": {k: v["raw"] for k, v in final_state["round_2_responses"].items()},
            "judge_reasoning": judge_text
        }
        
        return final_state["final_action"], json.dumps(full_log)

    def get_graph(self):
        try:
            self.app.get_graph().print_ascii()
        except:
            pass
