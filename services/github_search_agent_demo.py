import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from services.tavily_search import tavily_client
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
import os

load_dotenv()



web_search_agent_model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    # reasoning_format="parsed",
)



def search_github(topic, language="Python", max_results=2):
    """
    Search GitHub for repositories related to a topic.
    
    Args:
        topic (str): The topic or keyword to search.
        language (str): Programming language filter (default: Python).
        max_results (int): Maximum number of repos to return.

    Returns:
        list: A list of dictionaries with repo info (name, url, description).
    """
    url = "https://api.github.com/search/repositories"
    query = f"{topic} language:{language}"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.json().get('message')}")
        return []
    

    results = response.json().get("items", [])
    repos = []
    cnt = 0
    for repo in results:
        cnt += 1
        repos.append({
            "name": repo["full_name"],
            "url": repo["html_url"],
            "description": repo["description"]
        })
        if cnt == 5: break
    

    urls = []
    # print(repos)
    for r in repos:
        # print(r["name"], "-", r["url"])
        urls.append(r['url'])



    contents = []

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                tag.decompose()

            paragraphs = soup.find_all("p")

            text = " ".join(
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 50
            )

            if text:
                contents.append(text)

        except Exception as e:
            print(f"Skipping {url} | Reason: {e}")
        

    # print("\n \n ")
    # print(contents)

    res = "".join(contents)
    return res


search_github_tool = Tool(
    name="search_github",
    func=search_github,
    description="Search GitHub for Python repositories related to a topic and return their text content"
)


  



github_agent = create_agent(
    model=web_search_agent_model,
    tools=[search_github_tool],
    system_prompt="""
You are a professional github research agent.

Your goal:
- Use the search_github tool to gather accurate, up-to-date information
- If needed, call the tool multiple times
- Verify information consistency
- Remove noise, ads, and irrelevant content
- Return a clean, deep-research-ready content
""",
)


# if __name__ == "__main__":
#     result = github_agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": (
#                         "Research the following topic thoroughly:\n\n"
#                         "Retrieval Augmented Generation (RAG) for LLMs\n\n"
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

