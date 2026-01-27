import json
import subprocess
from fastmcp import FastMCP
from typing import List, Dict, Any
from services.mco.github_search import search_github_repos, clone_repo

# -----------------------------
# Load tools metadata (optional)
# -----------------------------
TOOLS_FILE = "services/mco/tools.json"
with open(TOOLS_FILE) as f:
    TOOLS = json.load(f)["tools"]

# Create the FastMCP server
mcp = FastMCP(name="Remote GitHub Grep MCP Server")

def grep_repo_for_code(repo_path: str, query: str, max_results: int) -> List[str]:
    """Return the matching code lines from a repo."""
    cmd = [
        "rg",  # ripgrep
        query,
        repo_path,
        "--line-number",
        "--no-heading",
        "--max-count",
        str(max_results)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    code_snippets = []
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            code_snippets.append(parts[2].strip())
    return code_snippets

@mcp.tool
def remote_grep(
    query: str,
    language: str = "python",
    max_repos: int = 3,
    max_results: int = 20
) -> List[Dict[str, Any]]:
    """
    Search GitHub repos for a query and return code snippets directly.
    """
    results: List[Dict[str, Any]] = []
    repos = search_github_repos(query, language, max_repos)

    for repo_url in repos:
        repo_path = clone_repo(repo_url)
        snippets = grep_repo_for_code(repo_path, query, max_results)
        for snippet in snippets:
            results.append({
                "repository": repo_url,
                "code": snippet
            })
    
    # Limit total results across all repos
    return results[:max_results]

if __name__ == "__main__":
    mcp.run(host="127.0.0.1", port=9000, transport="http")
