from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from ai.tools import get_tools
from ai.config import get_llm

llm = get_llm()
tools = get_tools()

SYSTEM_PROMPT = """
You are a retail intelligence assistant.

You help analyze retail data using available tools.

You MUST:
- Use tools when needed
- Give clear business insights
- Explain reasons
- Suggest actionable recommendations

Always respond in this format:

Insights:
- ...

Reasons:
- ...

Recommendations:
- ...
"""

agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    messages_modifier=SystemMessage(content=SYSTEM_PROMPT)  # ← try this
)


def run_agent(query: str):
    try:
        response = agent_executor.invoke({
            "messages": [("user", query)]
        })

        messages = response.get("messages", [])

        if not messages:
            return "No response generated."

        last_message = messages[-1]

        if hasattr(last_message, "content"):
            return last_message.content

        return str(last_message)

    except Exception as e:
        return f"Agent error: {str(e)}"