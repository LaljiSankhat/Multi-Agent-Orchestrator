import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

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



l = search_github_files("tool calling")

for r in l:
    print("\n", r)