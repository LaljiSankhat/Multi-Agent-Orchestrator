import os
import subprocess
import requests
from typing import List

GITHUB_API = "https://api.github.com/search/repositories"
CACHE_DIR = "repo_cache"

os.makedirs(CACHE_DIR, exist_ok=True)


def search_github_repos(query: str, language: str, max_repos: int) -> List[str]:
    q = f"{query} language:{language}"
    params = {"q": q, "sort": "stars", "order": "desc", "per_page": max_repos}

    res = requests.get(GITHUB_API, params=params)
    res.raise_for_status()

    items = res.json()["items"]
    return [repo["clone_url"] for repo in items]


def clone_repo(clone_url: str) -> str:
    repo_name = clone_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(CACHE_DIR, repo_name)

    if not os.path.exists(repo_path):
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, repo_path],
            check=True
        )

    return repo_path
