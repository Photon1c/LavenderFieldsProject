import anthropic
from anthropic import Anthropic
import os
import json
from typing import List, Dict, Any, Callable

class ClaudeSubagent:
    def __init__(self, name: str, system_prompt: str, model: str = "claude-3-7-sonnet-20250219"):
        """Initialize a Claude subagent with a specific role and system prompt."""
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.conversation_history = []
    
    def think(self, query: str) -> str:
        """Internal reasoning step - more verbose thought process."""
        messages = [
            {
                "role": "user",
                "content": f"THINK STEP: {query}\n\nPlease reason through this step by step before providing your final answer."
            }
        ]
        
        if self.conversation_history:
            messages = self.conversation_history + messages
            
        response = self.client.messages.create(
            model=self.model,
            system=f"{self.system_prompt}\nYou are the {self.name} subagent. Your role is to think deeply and show your reasoning.",
            messages=messages,
            max_tokens=1000
        )
        
        return response.content[0].text
    
    def act(self, query: str, include_history: bool = True) -> str:
        """Perform the subagent's main action and return a result."""
        messages = [{"role": "user", "content": query}]
        
        if include_history and self.conversation_history:
            messages = self.conversation_history + messages
            
        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt,
            messages=messages,
            max_tokens=1000
        )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response.content[0].text})
        
        return response.content[0].text

class Orchestrator:
    def __init__(self):
        """Initialize the orchestrator that manages multiple subagents."""
        self.subagents: Dict[str, ClaudeSubagent] = {}
        self.workflows: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_subagent(self, name: str, system_prompt: str, model: str = "claude-3-7-sonnet-20250219") -> None:
        """Add a new subagent to the orchestrator."""
        self.subagents[name] = ClaudeSubagent(name, system_prompt, model)
    
    def define_workflow(self, workflow_name: str, steps: List[Dict[str, Any]]) -> None:
        """Define a multi-step workflow using different subagents."""
        self.workflows[workflow_name] = steps
    
    def run_workflow(self, workflow_name: str, initial_input: str) -> Dict[str, str]:
        """Execute a predefined workflow with the given input."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found.")
        
        workflow = self.workflows[workflow_name]
        results = {"initial_input": initial_input}
        current_input = initial_input
        
        for step in workflow:
            agent_name = step["agent"]
            action = step.get("action", "act")
            
            if agent_name not in self.subagents:
                raise ValueError(f"Subagent '{agent_name}' not found.")
            
            agent = self.subagents[agent_name]
            
            # Format the input if a template is provided
            if "input_template" in step:
                formatted_input = step["input_template"].format(
                    input=current_input, 
                    **{k: v for k, v in results.items()}
                )
            else:
                formatted_input = current_input
            
            # Execute the appropriate action
            if action == "think":
                result = agent.think(formatted_input)
            else:  # Default to "act"
                result = agent.act(formatted_input)
            
            # Store the result
            results[f"{agent_name}_{action}"] = result
            
            # Update current input if this step's output should be passed to the next step
            if step.get("pass_output", True):
                current_input = result
        
        return results

# Example usage
def create_research_team() -> Orchestrator:
    """Create a research team with specialized subagents."""
    orchestrator = Orchestrator()
    
    # Add specialized subagents
    orchestrator.add_subagent(
        "researcher",
        "You are a thorough researcher who finds and organizes information effectively."
    )
    
    orchestrator.add_subagent(
        "analyzer",
        "You are an analytical assistant who examines information critically and identifies patterns."
    )
    
    orchestrator.add_subagent(
        "writer",
        "You are a skilled writer who creates clear, concise summaries and reports."
    )
    
    # Define a research workflow
    orchestrator.define_workflow(
        "research_topic",
        [
            {
                "agent": "researcher",
                "action": "think",
                "input_template": "What are the key aspects of {input} that need to be researched?"
            },
            {
                "agent": "researcher",
                "action": "act",
                "input_template": "Research the following topic thoroughly: {input}"
            },
            {
                "agent": "analyzer",
                "action": "act",
                "input_template": "Analyze the following research results and identify key patterns:\n\n{researcher_act}"
            },
            {
                "agent": "writer",
                "action": "act",
                "input_template": "Create a concise report on {input} based on this analysis:\n\n{analyzer_act}"
            }
        ]
    )
    
    return orchestrator

# Example of how to use this framework
if __name__ == "__main__":
    # Set your API key in environment variables
    os.environ["ANTHROPIC_API_KEY"] = "your_api_key"
    
    # Create a research team
    team = create_research_team()
    
    # Run a research workflow
    results = team.run_workflow("research_topic", "The impact of artificial intelligence on software development")
    
    # Print the final report
    print("\n=== FINAL REPORT ===\n")
    print(results["writer_act"])
