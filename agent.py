from typing import TypedDict, Annotated, List, Literal
import operator
import os
import json
import uuid
import asyncio
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from services.web_search_agent_demo import web_agent
from services.github_search_agent_demo import github_agent
from services.db_query_demo import postgres_agent, get_pg_connection
from services.orc_demo import orchestrator_model, orchestrator_prompts
from services.make_pretty_output import pretty

load_dotenv()



class AgentState(TypedDict):
    content_to_research: str
    research_content: Annotated[List[str], operator.add]
    node_to_call: List[Literal["web", "github", "db"]]
    final_research_summary: str
    approval: bool
    db_titles: List[str]



def ask_topic_node(state: AgentState):
    topic = interrupt({"question": "Enter research topic"})

    if topic.lower() == "exit":
        return Command(goto=END)

    return {
        "content_to_research": topic,
        "research_content": [],
        "node_to_call": [],
        "final_research_summary": "",
        "approval": None,
        "db_titles": [],
    }


def fetch_db_titles(state: AgentState):
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("SELECT title FROM documents")
    titles = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"db_titles": titles}


async def orchestrator_node(state: AgentState):
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
    
    print(f"Orchestrator have decided to do following \n {l} \n")
    
    state['node_to_call'] = l
    return state


def assign_workers(state: AgentState):
    sends = []
    for node in state["node_to_call"]:
        if node == "web":
            sends.append(Send("web_search", state))
        elif node == "github":
            sends.append(Send("github_search", state))
        elif node == "db":
            sends.append(Send("db_search", state))
    return sends



async def web_search_node(state: AgentState):

    print("\n Doing web search... \n")

    result = await web_agent.ainvoke(
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

    print(" \n Web search Completed \n")
    return {"research_content": [pretty(result)]}



async def github_search_node(state: AgentState):

    print("\n Doing github search... \n")

    result = await github_agent.ainvoke(
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

    print(" \n Github search Completed \n")


    return {"research_content": [pretty(result)]}


async def db_search_node(state: AgentState):

    print("\n Searching in DB... \n ")

    result = await postgres_agent.ainvoke(
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

    print("\n DB search Completed... \n ")

    return {"research_content": [pretty(result)]}


def join_results_node(state: AgentState):
    return state


async def final_summary_node(state: AgentState):
    print("\n Deep thinking on content... \n")
    combined = "\n\n".join(state["research_content"])
    response = await orchestrator_model.ainvoke(
        f"Summarize into a final research report:\n{combined}"
    )

    print(f"\n\n Final Summary \n \n {response.content} \n \n")
    return {"final_research_summary": response.content}


def approval_node(state: AgentState):
    decision = interrupt({
        "question": "Do you approve this research?",
        "options": ["yes", "no"]
    })
    return {"approval": decision == "yes"}



def save_db_node(state: AgentState):
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (title, content) VALUES (%s, %s)",
        (state["content_to_research"], state["final_research_summary"])
    )
    conn.commit()
    cur.close()
    conn.close()
    return state



graph = StateGraph(AgentState)

graph.add_node("ask_topic", ask_topic_node)
graph.add_node("fetch_db_titles", fetch_db_titles)
graph.add_node("orchestrator", orchestrator_node)

graph.add_node("web_search", web_search_node)
graph.add_node("github_search", github_search_node)
graph.add_node("db_search", db_search_node)

graph.add_node("join_results", join_results_node)
graph.add_node("final_summary", final_summary_node)
graph.add_node("approval", approval_node)
graph.add_node("save_db", save_db_node)

graph.add_edge(START, "ask_topic")
graph.add_edge("ask_topic", "fetch_db_titles")
graph.add_edge("fetch_db_titles", "orchestrator")

graph.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["web_search", "github_search", "db_search"]
)

graph.add_edge("web_search", "join_results")
graph.add_edge("github_search", "join_results")
graph.add_edge("db_search", "join_results")

graph.add_edge("join_results", "final_summary")
graph.add_edge("final_summary", "approval")

graph.add_conditional_edges(
    "approval",
    lambda state: state["approval"],
    {True: "save_db", False: "ask_topic"}
)

graph.add_edge("save_db", "ask_topic")



CHECKPOINTER_DB_URL = os.getenv("CHECKPOINTER_DB_URL")

async def main():
    async with AsyncPostgresSaver.from_conn_string(CHECKPOINTER_DB_URL) as memory:
        # await memory.setup()

        while True:
            workflow = graph.compile(checkpointer=memory)

            config = {
                "configurable": {
                    "thread_id": f"research-{uuid.uuid4()}"
                }
            }

            await workflow.ainvoke(
                {
                    "content_to_research": "",
                    "research_content": [],
                    "node_to_call": [],
                    "final_research_summary": "",
                    "approval": None,
                    "db_titles": [],
                },
                config
            )

            while True:
                snapshot = await workflow.aget_state(config)

                if not snapshot.interrupts:
                    break

                interrupt_data = snapshot.interrupts[0].value
                print("\n" + interrupt_data["question"])

                if "options" in interrupt_data:
                    print("Options:", interrupt_data["options"])

                user_input = input("> ").strip()

                if user_input.lower() == "exit":
                    print("\n Exiting research assistant")
                    return

                await workflow.ainvoke(
                    Command(resume=user_input),
                    config
                )

            print("\n Research cycle completed\n")

if __name__ == "__main__":
    asyncio.run(main())
