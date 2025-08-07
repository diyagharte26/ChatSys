from typing import Dict, Any, List
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
        
        system_prompt = """You are KubeSage, an AI assistant specialized in Kubernetes operations and automation.

You have access to various Kubernetes tools that can help users manage their clusters, deploy applications, troubleshoot issues, and automate workflows.

When a user asks for help with Kubernetes:
1. Analyze their request carefully
2. Choose the most appropriate tool(s) to help them
3. Execute the tool(s) with the correct parameters
4. Provide clear, helpful explanations of what you did and the results

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
        
        if not messages or len(messages) == 0:
            return AgentResponse(content="I received an empty message list.", tools_used=[])

        # The user's latest message is the primary input.
        user_input = messages[-1]["content"]
        
        # The rest of the conversation is the history.
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] == "system":
                chat_history.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
        
        try:
            result = await self.agent_executor.ainvoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            tools_used = self._extract_tools_used(result)
            
            return AgentResponse(
                content=result.get("output", "I could not generate a response."),
                tools_used=tools_used
            )
            
        except Exception as e:
            # Provide a more detailed error message for debugging.
            error_content = f"I encountered an error while processing your request. This can sometimes be caused by an issue with the LLM provider's API or content filters. Details: {str(e)}"
            return AgentResponse(
                content=error_content,
                tools_used=[]
            )
    
    def _extract_tools_used(self, result: Dict[str, Any]) -> List[str]:
        """Extract list of tools used from agent result"""
        tools_used = []
        
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if hasattr(step, 'tool') and step.tool:
                    tools_used.append(step.tool)
                elif len(step) >= 2 and hasattr(step[0], 'tool'):
                    tools_used.append(step[0].tool)
        
        return list(set(tools_used))

def get_agent(llm_adapter) -> KubeSageAgent:
    """Factory function to create agent instance"""
    return KubeSageAgent(llm_adapter)

# The SimpleAgent class remains unchanged
class SimpleAgent:
    """Fallback simple agent without LangChain dependencies"""
    
    def __init__(self, llm_adapter):
        self.llm_adapter = llm_adapter
        self.tools = k8s_tools
    
    async def process_messages(self, messages: List[Dict[str, Any]]) -> AgentResponse:
        """Simple processing without tool selection"""
        
        user_input = messages[-1]["content"] if messages else ""
        
        selected_tool = self._select_tool_by_keywords(user_input)
        
        if selected_tool:
            try:
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
                for tool in self.tools:
                    if hasattr(tool, 'name') and tool.name == tool_name:
                        return tool
        
        return None 