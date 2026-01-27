from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from services.tavily_search import tavily_client
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os

load_dotenv()


web_search_agent_model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    # reasoning_format="parsed",
)


@tool
def web_search(research_content: str):
    """Tool to do web search based on the research content given by user"""
    response = tavily_client.search(
        query=research_content,
        search_depth="advanced",      
        include_raw_content=False,    
        # max_results=5
    )
    combined_content = ""

    for r in response['results']:
        combined_content += r['content']
    
        
    return combined_content
    



web_agent = create_agent(
    model=web_search_agent_model,
    tools=[web_search],
    system_prompt="""
You are a professional web research agent.

Your goal:
- Use the web_search tool to gather accurate, up-to-date information
- If needed, call the tool multiple times
- Verify information consistency
- Remove noise, ads, and irrelevant content
- Return a clean, deep-research-ready content
""",
)


# if __name__ == "__main__":
#     result = web_agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": (
#                         "Research the following topic thoroughly:\n\n"
#                         "Latest advancements in Retrieval Augmented Generation (RAG) for LLMs\n\n"
#                         "Instructions:\n"
#                         "- Prefer authoritative and recent sources\n"
#                         "- Merge overlapping information\n"
#                         "- Keep the output factual"
#                     ),
#                 }
#             ]
#         }
#     )

#     # print("\nFINAL OUTPUT:\n")
#     # print(result)

#     print("\n \n")

#     for i, msg in enumerate(result["messages"], 1):
#         print(f"\n--- Step {i} | {msg.__class__.__name__} ---")

#         if isinstance(msg, HumanMessage):
#             print("USER:\n", msg.content)

#         elif isinstance(msg, ToolMessage):
#             print(f"TOOL [{msg.name}]:\n", msg.content[:800], "...\n")

#         elif isinstance(msg, AIMessage):
#             if msg.tool_calls:
#                 print("AI decided to call tools:")
#                 for tc in msg.tool_calls:
#                     print(f" - {tc['name']}({tc['args']})")
#             else:
#                 print("AI FINAL OUTPUT:\n", msg.content)
    

    
#     tool_calls = [
#         m for m in result["messages"]
#         if isinstance(m, ToolMessage)
#     ]

#     print("Total tool calls:", len(tool_calls))

#     print("\n \n")























# web_search_agent_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a professional web research agent.

#             Your goal:
#             - Use the web_search tool to gather accurate, up-to-date information
#             - If needed, call the tool multiple times
#             - Verify information consistency
#             - Remove noise, ads, and irrelevant content
#             - Return a clean, research-ready summary
#             """
#         ),
#         (
#             "human",
#             """
#             Research the following topic thoroughly:

#             {research_content}

#             Instructions:
#             - Prefer authoritative and recent sources
#             - Merge overlapping information
#             - Keep the output factual and concise
#             """
#         ),
#     ]
# )