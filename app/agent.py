from typing import Dict, Any, List
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from .k8s_tools import k8s_tools
from .models import AgentResponse

class KubeSageAgent:
    """LangGraph-powered agent for Kubernetes operations"""
    
    def __init__(self, llm_adapter):
        self.llm_adapter = llm_adapter
        self.tools = k8s_tools
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create LangChain agent with Kubernetes tools"""
        
        # System prompt for Kubernetes operations
        system_prompt = """You are KubeSage, an AI assistant specialized in Kubernetes operations and automation.

You have access to various Kubernetes tools that can help users manage their clusters, deploy applications, troubleshoot issues, and automate workflows.

When a user asks for help with Kubernetes:
1. Analyze their request carefully
2. Choose the most appropriate tool(s) to help them
3. Execute the tool(s) with the correct parameters
4. Provide clear, helpful explanations of what you did and the results

Available tools:
{tools}

Guidelines:
- Always prioritize safety and security
- Explain what you're doing before executing dangerous operations
- Ask for confirmation for destructive operations
- Provide kubectl commands as alternatives when appropriate
- Help users understand Kubernetes concepts along the way

Be concise but thorough in your responses. Always aim to be helpful and educational.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_tools_agent(
            llm=self.llm_adapter.get_langchain_llm(),
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    async def process_messages(self, messages: List[Dict[str, Any]]) -> AgentResponse:
        """Process conversation messages and return response"""
        
        # Extract the latest user message
        user_input = ""
        chat_history = []
        
        for msg in messages:
            if msg["role"] == "user":
                user_input = msg["content"]
            elif msg["role"] == "system":
                chat_history.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Convert to appropriate message type for chat history
                pass
        
        # Execute agent
        try:
            result = await self.agent_executor.ainvoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Extract tools used
            tools_used = self._extract_tools_used(result)
            
            return AgentResponse(
                content=result["output"],
                tools_used=tools_used
            )
            
        except Exception as e:
            return AgentResponse(
                content=f"I encountered an error while processing your request: {str(e)}",
                tools_used=[]
            )
    
    def _extract_tools_used(self, result: Dict[str, Any]) -> List[str]:
        """Extract list of tools used from agent result"""
        tools_used = []
        
        # Check if intermediate steps contain tool usage
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if hasattr(step, 'tool') and step.tool:
                    tools_used.append(step.tool)
                elif len(step) >= 2 and hasattr(step[0], 'tool'):
                    tools_used.append(step[0].tool)
        
        return list(set(tools_used))  # Remove duplicates

def get_agent(llm_adapter) -> KubeSageAgent:
    """Factory function to create agent instance"""
    return KubeSageAgent(llm_adapter)

class SimpleAgent:
    """Fallback simple agent without LangChain dependencies"""
    
    def __init__(self, llm_adapter):
        self.llm_adapter = llm_adapter
        self.tools = k8s_tools
    
    async def process_messages(self, messages: List[Dict[str, Any]]) -> AgentResponse:
        """Simple processing without tool selection"""
        
        # Get the last user message
        user_input = messages[-1]["content"] if messages else ""
        
        # Simple keyword-based tool selection
        selected_tool = self._select_tool_by_keywords(user_input)
        
        if selected_tool:
            try:
                # Execute the tool (simplified)
                tool_result = "Tool execution simulated"
                return AgentResponse(
                    content=f"I used the {selected_tool.name} tool. {tool_result}",
                    tools_used=[selected_tool.name]
                )
            except Exception as e:
                return AgentResponse(
                    content=f"Error executing tool: {str(e)}",
                    tools_used=[]
                )
        else:
            # Use LLM for general response
            try:
                llm_response = await self.llm_adapter.call_raw(messages)
                content = llm_response.get("content", "I'm not sure how to help with that.")
                return AgentResponse(
                    content=content,
                    tools_used=[]
                )
            except Exception as e:
                return AgentResponse(
                    content=f"I encountered an error: {str(e)}",
                    tools_used=[]
                )
    
    def _select_tool_by_keywords(self, user_input: str):
        """Simple keyword-based tool selection"""
        user_input_lower = user_input.lower()
        
        # Map keywords to tools
        keyword_map = {
            "pods": "get_pods",
            "services": "get_services",
            "deployments": "get_deployments",
            "logs": "get_logs",
            "describe": "describe_resource",
            "create": "create_resource",
            "delete": "delete_resource",
            "scale": "scale_deployment"
        }
        
        for keyword, tool_name in keyword_map.items():
            if keyword in user_input_lower:
                # Find the tool by name
                for tool in self.tools:
                    if hasattr(tool, 'name') and tool.name == tool_name:
                        return tool
        
        return None