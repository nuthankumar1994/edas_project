"""
Main application for LangGraph Agent System
"""
import logging
from src.agents.coordinator_agent import RAGAgentManager as AgentManager
from src.core import ChatRequest
from langchain_core.messages import AIMessage, HumanMessage
agent_manager = AgentManager()
from datetime import datetime, date
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import argparse
import json
import sys
from src.utils.loggers import logger, timeit


def agent_chat(request: ChatRequest):
    """
    Chat with an agent.

    Args:
        request: Chat request with message and agent metadata

    Returns:
        Dict with status and agent response
    """
    try:
        # --- Basic request validation ---
        if not request.message:
            return {
                "status": "failure",
                "error": "Message cannot be empty."
            }

        agent_manager.initialize()

        result = agent_manager.process_message(
            message=request.message,
            thread_id=request.thread_id,
            user_id=request.user_id,
        ) or {}

        agent_result = result.get("result") or {}
        messages = agent_result.get("messages")

        if not messages:
            return {
                "status": "failure",
                "error": "No messages returned from agent."
            }

        last_message = messages[-1]

        content = getattr(last_message, "content", None)
        if not isinstance(content, str):
            return {
                "status": "failure",
                "error": "Agent returned a message without textual content."
            }

        normalized_content = content.replace("\n\n", "\n")
        return {
            "status": "success",
            "result": normalized_content,
        }

    except Exception as e:
        logger.exception(f"agent_chat failed {e}")  # preserves traceback
        return {
            "status": "failure",
            "error": f"Internal server error. {e}"
        }
    
def get_application():
    _app = FastAPI(
        title=os.environ['app_name'], description="aws_ai_agent", version="0.0.1", debug=True
    )
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.environ['cors_allow_origins']],
        allow_credentials=os.environ['cors_allow_credentials'],
        allow_methods=[os.environ['cors_allow_methods']],
        allow_headers=[os.environ['cors_allow_headers']],
    )
    return _app

app = get_application() 
prefix_router = APIRouter(prefix="/api/v1/docassistagent")


@prefix_router.post("/agent_chat")
def chat(request: ChatRequest):
    res = agent_chat(request)
    return res

@prefix_router.post(
    "/redis_flushdb", response_model_exclude_none=True
)
def redis_flushdb(): 
    try:
        from redis import Redis
        redis_host = os.getenv("VALKEY_HOST", "localhost")
        redis_socket_timeout = 300
        rq_conn = Redis(host=redis_host, socket_timeout=redis_socket_timeout)
        rq_conn.flushdb()
        return {"status": "success"}
    except Exception as e:
        return {"status": "failure", "error": str(e)}

app.include_router(prefix_router)


import shutil
from colorama import Fore, Style, init

init(autoreset=True)

def format_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def bubble(text, sender="user"):
    width = shutil.get_terminal_size().columns - 4
    wrapped = []
    line = ""

    for word in text.split():
        if len(line) + len(word) + 1 <= width:
            line += (" " if line else "") + word
        else:
            wrapped.append(line)
            line = word
    wrapped.append(line)

    bubble_lines = []
    for i, w in enumerate(wrapped):
        prefix = "│ " if i > 0 else "┌ "
        suffix = " │" if i < len(wrapped)-1 else " ┘"
        bubble_lines.append(prefix + w + suffix)

    return "\n".join(bubble_lines)



def interactive_mode():
    print(Fore.CYAN + "Entering interactive mode. Type '/exit' to quit.")
    thread_id = str(uuid.uuid4())
    user_id = "1"

    print(Fore.BLUE + f"User: {user_id} | Thread: {thread_id}")

    while True:
        try:
            user_input = input(Fore.YELLOW + "\nYou > " + Style.RESET_ALL).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + Fore.RED + "Exiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", ":q", "quit", "exit"):
            print(Fore.RED + "Goodbye!")
            break

        # Format user's bubble
        print(Fore.GREEN + f"\n[{format_timestamp()}] You:")
        print(Fore.GREEN + bubble(user_input, sender="user"))

        # prepare chat request
        request = ChatRequest(
            message=user_input,
            thread_id=thread_id,
            user_id=user_id
        )

        resp = agent_chat(request)
        if resp['status'] == "success":
            assistant_msg = resp["result"]
        else:
            assistant_msg = "error in code" + resp["error"]
        print(Fore.CYAN + f"\n[{format_timestamp()}] Assistant:")
        print(Fore.CYAN + bubble(assistant_msg, sender="assistant"))


def main():
    interactive_mode()


if __name__ == "__main__":
    main() 