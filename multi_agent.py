from typing import TypedDict, Annotated, List, Literal
import os
import operator
import asyncio
from dotenv import load_dotenv
from unittest import result
from langchain_groq import ChatGroq
from langgraph.types import interrupt, Command, Send
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from services.web_search_agent_demo import web_search, web_search_agent_model, web_agent
from services.github_search_agent_demo import search_github, search_github_tool, github_agent
from services.db_query_demo import db_model, postgres_agent, search_postgres, get_pg_connection
from services.orc_demo import orchestrator_model, orchestrator_prompts
from services.make_pretty_output import pretty
from services.text_splitter import split_text_into_chunks

load_dotenv()






class AgentState(TypedDict):
    content_to_research: str

    research_content: Annotated[List[str], operator.add]
    node_to_call: List[Literal['web', 'github', 'db']]

    final_research_summary: str
    approval: bool

    db_titles: List[str]


def ask_topic_node(state: AgentState):

    print("What type of content you want to research today ? \n")
    user_input = input("give me topic: ")

    state['content_to_research'] = user_input
    new_state = {
        "content_to_research": user_input,
        "research_content": None,
        "node_to_call": None,
        "final_research_summary": None,
        "approval": None,
        "db_titles": state['db_titles']
    }

    return new_state


def orchestrator_node(state: AgentState):

    response = orchestrator_model.invoke(
        orchestrator_prompts.format_messages(
            user_content=state['content_to_research'],
            db_content=state['db_titles']
        )
    )

    ans = response.content
    l = []

    for a in ans.split(','):
        l.append(a.split("'")[1])
    
    state['node_to_call'] = l
    return state


def web_search_node(state: AgentState):

    result = web_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Research the following topic thoroughly:\n\n"
                        f"{state['content_to_research']}\n\n"
                        "Instructions:\n"
                        "- Prefer authoritative and recent sources\n"
                        "- Merge overlapping information\n"
                        "- Keep the output factual"
                    ),
                }
            ]
        }
    )

    final_output = pretty(result)

    return {"research_content": [final_output]}


def github_search_node(state: AgentState):

    result = github_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Research the following topic thoroughly:\n\n"
                        f"{state['content_to_research']}\n\n"
                        "Instructions:\n"
                        "- Use github contents only \n"
                        "- Merge overlapping information\n"
                        "- Keep the output factual"
                    ),
                }
            ]
        }
    )


    final_output = pretty(result)

    return {"research_content": [final_output]}


def db_query_node(state: AgentState):

    result = postgres_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Research the following topic thoroughly:\n\n"
                        f"{state['content_to_research']}\n\n"
                        "Instructions:\n"
                        "- Use database information only\n"
                        "- Merge overlapping results\n"
                        "- Keep the output factual and concise"
                    ),
                }
            ]
        }
    )

    final_output = pretty(result)

    return {"research_content": [final_output]}


def final_summary_node(state: AgentState):
    # Combine all research gathered
    combined_content = "\n\n".join(state['research_content'])
    
    response = orchestrator_model.invoke(
        f"Summarize the following research into a final report:\n{combined_content}"
    )
    
    return {"final_research_summary": response.content}


def approval_node(state: AgentState):

    decision = interrupt(
        {
            'question': "would you approve this ?",
            "options": ["yes", "no"]
        }
    )

    return {"approval": True if decision == "yes" else False}

def assign_workers(state: AgentState):
    sends = []

    for node in state['node_to_call']:
        if node == "web":
            sends.append(Send("web_search", state))
        elif node == "github":
            sends.append(Send("github_search", state))
        elif node == "db":
            sends.append(Send("db_search", state))

    return sends

def save_db_node(state: AgentState):

    conn = get_pg_connection()
    cur = conn.cursor()


    query = """
    INSERT INTO documents (title, content)
    VALUES (%s, %s)
    """

    values = (state['content_to_research'], state['final_research_summary'])

    cur.execute(query, values)

    conn.commit()   
    cur.close()
    conn.close()

    return state




graph = StateGraph(AgentState)


# adding nodes in graph 
graph.add_node("ask_topic", ask_topic_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("web_search", web_search_node)
graph.add_node("github_search", github_search_node)
graph.add_node("db_search", db_query_node)
graph.add_node("final_summary", final_summary_node)
graph.add_node("approval", approval_node)
graph.add_node("save_db", save_db_node)


graph.add_edge(START, "ask_topic")
graph.add_edge("ask_topic", "orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["web_search", "github_search", "db_search"]
)

graph.add_edge("web_search", "final_summary")
graph.add_edge("github_search", "final_summary")
graph.add_edge("db_search", "final_summary")

graph.add_edge("final_summary", "approval")
graph.add_conditional_edges(
    "approval",
    lambda state: state["approval"],
    {
        True: "save_db",
        False: "ask_topic",
    }
)
graph.add_edge("save_db", "ask_topic")


config = {
    "configurabel": {
        "thread_id": "multiagent_id_1"
    }
}



intial_state = {
    "content_to_research": "",
    "research_content": [],
    "node_to_call": [],
    "final_research_summary": "",
    "approval": None,
    "db_titles": []
}



CHECKPOINTER_DB_URL = os.getenv("CHECKPOINTER_DB_URL")

async def main():
    async with AsyncPostgresSaver.from_conn_string(CHECKPOINTER_DB_URL) as memory:

        # run only one time to create database tables``
        # await memory.setup()

        workflow = graph.compile(
            checkpointer=memory
        )
        print("AI Research Agent: Hello! What topic would you like to research today?")


        while True:
            # user_topic = input("\nEnter topic (or exit): ").strip()
            # if user_topic.lower() in ["exit", "quit"]:
            #     break

            await workflow.ainvoke(
                {
                    "content_to_research": "",
                    "research_content": [],
                    "node_to_call": [],
                    "final_research_summary": "",
                    "approval": None,
                    "db_titles": []
                },
                config
            )


            while True:
                snapshot = await workflow.aget_state(config)

                if not snapshot.interrupts:
                    break

                interrupt_data = snapshot.interrupts[0].value
                print("\n", interrupt_data["question"])
                print("Options:", interrupt_data["options"])

                user_input = input("> ").strip().lower()

                # If refinement needed, ask sub-topic
                if interrupt_data["question"].startswith("Do you want to research") and user_input == "yes":
                    sub_topic = input("\nEnter specific sub-topic: ").strip()
                    await workflow.ainvoke(
                        Command(resume=user_input, update={"userInterest": sub_topic, "userRelatedResearch": None}),
                        config
                    )
                else:
                    await workflow.ainvoke(Command(resume=user_input), config)

            print("\n Research cycle completed ...")

