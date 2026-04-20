from langgraph.prebuilt import create_react_agent
from ai.tools import get_tools
from ai.config import get_llm

llm = get_llm()
tools = get_tools()

SYSTEM_PROMPT = """
You are a retail intelligence assistant.

You have access to tools for:
- customer segmentation
- sales trend analysis
- product association analysis

Answer the user's question by:
1. Using tools when needed
2. Giving clear insights
3. Explaining reasons
4. Suggesting actions

Respond in structured format:
- Insights
- Reasons
- Recommendations
"""

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT
)

def run_agent(query: str):
    response = agent_executor.invoke({
        "messages": [("user", query)]
    })
    return response["messages"][-1].content