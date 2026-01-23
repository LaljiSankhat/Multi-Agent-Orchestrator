
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def pretty(result):
    for i, msg in enumerate(result["messages"], 1):
        print(f"\n--- Step {i} | {msg.__class__.__name__} ---")

        if isinstance(msg, HumanMessage):
            print("USER:\n", msg.content)

        elif isinstance(msg, ToolMessage):
            print(f"TOOL [{msg.name}]:\n", msg.content[:800], "...\n")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print("AI decided to call tools:")
                for tc in msg.tool_calls:
                    print(f" - {tc['name']}({tc['args']})")
            else:
                print("AI FINAL OUTPUT:\n", msg.content)

    tool_calls = [
        m for m in result["messages"]
        if isinstance(m, ToolMessage)
    ]

    print("\nTotal tool calls:", len(tool_calls))
    print("\n\n")


    return msg.content