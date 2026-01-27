from typing import List, Dict
import requests
import base64
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import time

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")  # optional for higher rate limit

# -------------------------
# LLM MODEL
# -------------------------
web_search_agent_model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=800,
)

# -------------------------
# CHUNKING FUNCTION        
# -------------------------
def chunk_text(text: str, chunk_size=1200, overlap=200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# -------------------------
# GITHUB SEARCH
# -------------------------
def search_github_repos(topic: str, language="Python", max_results=2) -> List[Dict]:
    url = "https://api.github.com/search/repositories"
    query = f"{topic} language:{language}"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print("GitHub search error:", e)
        return []

    repos = response.json().get("items", [])[:max_results]
    return [{"owner": r["owner"]["login"], "repo": r["name"], "default_branch": r["default_branch"]} for r in repos]

# -------------------------
# FETCH FILE CONTENT
# -------------------------
def fetch_file_content(owner: str, repo: str, path: str, branch="main") -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return ""

    data = r.json()
    content_base64 = data.get("content", "")
    if content_base64:
        return base64.b64decode(content_base64).decode("utf-8")
    return ""

# -------------------------
# RECURSIVE FETCH REPO FILES
# -------------------------
def fetch_repo_files(owner: str, repo: str, path="", branch="main") -> Dict[str, str]:
    """
    Recursively fetch all Python files in a repository.
    Returns {file_path: file_content}.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return {}

    items = r.json()
    if isinstance(items, dict) and items.get("type") == "file":
        content_base64 = items.get("content", "")
        if content_base64:
            return {items["path"]: base64.b64decode(content_base64).decode("utf-8")}
        return {}

    repo_files = {}
    for item in items:
        if item["type"] == "file" and item["name"].endswith(".py"):
            content = fetch_file_content(owner, repo, item["path"], branch)
            if content:
                repo_files[item["path"]] = content
        elif item["type"] == "dir":
            repo_files.update(fetch_repo_files(owner, repo, path=item["path"], branch=branch))
        time.sleep(0.2)  # avoid GitHub rate limit
    return repo_files

# -------------------------
# FETCH AND CHUNK REPO CODE
# -------------------------
def fetch_repo_code(owner: str, repo: str, branch: str) -> Dict[str, List[str]]:
    repo_files = fetch_repo_files(owner, repo, branch)
    repo_code = {}
    for fpath, content in repo_files.items():
        repo_code[fpath] = chunk_text(content)
    return repo_code

# -------------------------
# SUMMARIZE CODE CHUNKS
# -------------------------
def summarize_code_chunks(chunks: List[str]) -> str:
    summaries = []

    for i, chunk in enumerate(chunks, 1):
        response = web_search_agent_model.invoke(
            input=f"Summarize the following Python code:\n\n{chunk}"
        )
        summaries.append(response.content)
        print(f"Processed chunk {i}/{len(chunks)}")
        time.sleep(0.5)

    return "\n".join(summaries)

# -------------------------
# MAIN PIPELINE
# -------------------------
if __name__ == "__main__":
    topic = "Retrieval Augmented Generation (RAG) for LLMs"
    print(f"Searching GitHub for topic: {topic} ...")
    repos = search_github_repos(topic, max_results=2)

    final_results = []

    for repo_info in repos:
        owner = repo_info["owner"]
        repo = repo_info["repo"]
        branch = repo_info["default_branch"]
        print(f"\nFetching code for repo: {owner}/{repo} (branch: {branch})")

        repo_code = fetch_repo_code(owner, repo, branch)

        repo_summary = {}
        for fpath, chunks in repo_code.items():
            summary = summarize_code_chunks(chunks)
            repo_summary[fpath] = summary

        final_results.append({
            "repo": f"{owner}/{repo}",
            "code": repo_code,      # full chunked code for UI
            "summary": repo_summary # summaries per file for UI
        })

    # -------------------------
    # SHOW RESULTS
    # -------------------------
    for repo in final_results:
        print(f"\n=== Repository: {repo['repo']} ===")
        for fpath, code_chunks in repo["code"].items():
            print(f"\nFile: {fpath}")
            print("\n".join(code_chunks[:2]))  # first 2 chunks preview
        for fpath, summary in repo["summary"].items():
            print(f"\nSummary for {fpath}:\n{summary[:500]}...\n")  # preview
