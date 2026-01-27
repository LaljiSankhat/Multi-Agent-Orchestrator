import os
import psycopg2
from langchain.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import os

load_dotenv()

# Required env vars:
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname
# GROQ_API_KEY=your_groq_key



def get_pg_connection():
    return psycopg2.connect(os.getenv("DB_URL"))



@tool
def search_postgres(topic: str) -> str:
    """
    Search for documents related to a topic in PostgreSQL.
    """
    conn = get_pg_connection()
    cur = conn.cursor()

    query = """
    SELECT title, content
    FROM documents
    WHERE to_tsvector('english', content)
          @@ plainto_tsquery('english', %s)
    LIMIT 5;
    """

    cur.execute(query, (topic,))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        return "No relevant data found in the database."

    results = []
    for title, content in rows:
        results.append(
            f"Title: {title}\nContent: {content[:300]}"
        )

    return "\n\n".join(results)



db_model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
)


postgres_agent = create_agent(
    model=db_model,
    tools=[search_postgres],
    system_prompt="""
You are a professional database research agent.

Your goal:
- Given a topic, search it in the PostgreSQL database
- Use the search_postgres tool to retrieve relevant data
- Call the tool multiple times if needed
- Remove irrelevant or duplicate content
- Return a clean, factual, research-ready response
""",
)


# if __name__ == "__main__":

#     result = postgres_agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": (
#                         f"Research the following topic thoroughly:\n\n"
#                         f"vector search\n\n"
#                         "Instructions:\n"
#                         "- Use database information only\n"
#                         "- Merge overlapping results\n"
#                         "- Keep the output factual and concise"
#                     ),
#                 }
#             ]
#         }
#     )

#     print("\n\n")

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

#     print("\nTotal tool calls:", len(tool_calls))
#     print("\n\n")