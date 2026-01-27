# Testig Orchestrator node 




from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os

load_dotenv()





orchestrator_model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=None,
    # reasoning_format="parsed",
)

# orchestrator_prompts = ChatPromptTemplate.from_template(
#     """
#     You are a professional orchestrator.
#     you define what kind of search needed for given topic or content of the research.

#     USER_CONTENT (content to research):
#     {user_content}

#     DB_CONTENT (what are titles in db which has content):
#     {db_content}


#     Instructions:
#     - you are given three options to reply ['web', 'github', 'db'].
#     - you can give multiple also but make sure always give in the list.
#     - now you have to identify the USER_CONTENT topic and think about what task need to perform weather it web, git or db.
#     - you can give multiple task as answer also.
#     - for db you have provided DB_CONTENT which has title of the content present in db based on that you have to decide weather db query required or not.

#     Answer format:
#     - give answer only as a list no explanation is required
#     - answer always list like ['web', 'git']

    
#     """
# )

orchestrator_prompts = ChatPromptTemplate.from_template(
    """
    You are a professional research orchestrator. Your goal is to determine which search tools are required to gather comprehensive information on a content.

    USER_CONTENT (Content to research):
    {user_content}

    DB_CONTENT (Existing database titles):
    {db_content}

    Selection Criteria for Options:
    1. 'web': Use for general facts, news, official websites, and broad overviews.
    2. 'git': Use if the topic asked for code demos, public repositories or public github repositories files 
    3. 'db': Use ONLY if the content closely matches or overlaps with one of the titles in DB_CONTENT.

    Instructions:

    - Even for non-technical topics, think: "Is there likely a dataset or specialized tool for this on GitHub?" 
    - For example, a geography topic like "Mountains" would need 'web' for facts.

    Answer format:
    - Return ONLY a Python list of strings.
    - No explanation, no intro text.
    """
)


response = orchestrator_model.invoke(
    orchestrator_prompts.format_messages(
        user_content="Explain langchain",
        db_content=['Langchain with postgres', 'Vector search', 'Large language models']
    )
)

print(response.content)
print(type(response.content))

ans = response.content

l = []

for a in ans.split(','):
    print(a.split("'")[1])
    l.append(a.split("'")[1])



print(l)