import os
import requests
from dotenv import load_dotenv

from services.make_pretty_output import pretty

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool

load_dotenv()


def search_github_repositories(query: str):
    """
    Search for GitHub repositories related to a specific research topic.
    Returns a list of repositories with their descriptions and star counts.
    """
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc"
    headers = {
        "Authorization": f"token {os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    items = response.json().get('items', [])[:5]  # Get top 5 results
    results = []
    for item in items:
        results.append({
            "name": item['full_name'],
            "description": item['description'],
            "url": item['html_url'],
            "stars": item['stargazers_count']
        })
    
    # results will be List[dict]

    return results



def search_github_files(research_topic: str):
    """
    Finds code demos and Jupyter Notebooks for a research topic.
    Returns file paths and direct links to code examples.
    """
    query = f"{research_topic} extension:ipynb path:examples path:demo path:notebooks"
    url = f"https://api.github.com/search/code?q={query}"

    headers = {
        "Authorization": f"token {os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    items = response.json().get('items', [])[:5]  # Top 5 code files
    results = []
    for item in items:
        results.append({
            "repo": item['repository']['full_name'],
            "file_path": item['path'],
            "demo_url": item['html_url'].replace("/blob/", "/raw/"), # Link to raw code
            "web_url": item['html_url']
        })

    if not results:
        return "No specific demo files found. Try a broader search."

    # results will be List[dict]
    return results


def fetch_code(url: str):

    # url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/langchain/langchain/chains/base.py"

    resp = requests.get(url)


    nb = resp.json() 
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] in ("code", "markdown"):
            parts.append("".join(cell["source"]))
    code_text = "\n".join(parts)
    # code_text = resp.text
    # print(code_text)
    return code_text

# fetch_code("https://raw.githubusercontent.com/splunk/splunk-mltk-container-docker/efa5f31e0fd8896fd62928b5f0d4d20f726b9033/notebooks/stumpy.ipynb")



def code_generation(query: str):

    API_KEY = os.getenv("GROQ_CODE_GENERATION_API_KEY")
    model = "llama-3.3-70b-versatile"
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"{query}"}
        ]
    }

    resp = requests.post(url, headers=headers, json=data)
    # print(resp.json().content)
    data = resp.json()
    output = data["choices"][0]["message"]["content"]
    # print(output)
    return output


# code_generation("Write a C++ code for dijkstra algorithm")

web_search_agent_model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    # reasoning_format="parsed",
)



search_github_repo_tool = Tool(
    name="search_github_repo_tool",
    func=search_github_repositories,
    description="Search GitHub for Python repositories related to a content"
)

search_github_files_tool = Tool(
    name="search_github_files_tool",
    func=search_github_files,
    description="Search GitHub for Python files or notebooks related to a content"
)

code_generation_tool = Tool(
    name="code_generation_tool",
    func=code_generation,
    description="generate code based on the user queries"
)



github_agent = create_agent(
    model=web_search_agent_model,
    tools=[search_github_repo_tool, search_github_files_tool, code_generation_tool],
    system_prompt="""
You are a professional github research agent.

Your goal:
- Use the search_github_repo_tool, search_github_files_tool and code_generation_tool tool to gather accurate, up-to-date information
- If needed, call the tool multiple times
- Verify information consistency
- Remove noise, ads, and irrelevant content
- Return combined output of all tool call results as it is not as summary
""",
)





if __name__ == "__main__":
    result = github_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain LangChain. "
                        "First find top GitHub repositories. "
                        "Then find demo code or notebooks. "
                        "Finally generate a simple example."
                    ),
                }
            ]
        }
    )

    print(pretty(result))

    