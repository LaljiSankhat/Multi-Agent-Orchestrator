from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# response = tavily_client.search(
#     query="Artificial Intelligence and Recent Investment",
#     search_depth="advanced",      
#     include_raw_content=False,    
#     # max_results=5
# )


# combined_content = ""

# for r in response['results']:
#     combined_content += r['content']
   
    
# print(combined_content)

