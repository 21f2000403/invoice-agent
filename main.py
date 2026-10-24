# main.py

import os
import json
import uuid
import hashlib
import sqlite3
import threading

from typing import Any
from fastapi import FastAPI, Header, HTTPException, Body
from pydantic import BaseModel


app = FastAPI(
    title="A2A Invoice Agent",
    description="An agent for evidence-based invoice decisions",
    version="1.0.0",
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

A2A_TOKEN = os.getenv("A2A_TOKEN", "local-test-token")
DATABASE = os.getenv("DATABASE_PATH", "invoice_agent.db")

BATCH_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-claim-batch+json"
)
PROPOSALS_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-proposals+json"
)
RECEIPTS_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-receipts+json"
)
RESULTS_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-results+json"
)

# Protect SQLite task updates from concurrent requests
db_lock = threading.RLock()


class SendMessageBody(BaseModel):
    message: dict[str, Any]
    configuration: dict[str, Any] = {}


# --------------------------------------------------
# Database
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(
        DATABASE,
        timeout=15,
    )
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with db_lock:
        connection = get_connection()

        connection.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                context_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(principal, message_id)
            )
        """)

        connection.commit()
        connection.close()


initialize_database()


# --------------------------------------------------
# Basic routes
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "Invoice Agent is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/.well-known/agent-card.json")
def agent_card():
    # Set AGENT_BASE_URL to the exact public HTTPS base URL
    # before deployment. The local URL is for development only.
    base_url = os.getenv(
        "AGENT_BASE_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")

    return {
        "name": "Invoice Action Agent",
        "description": (
            "Analyzes invoice evidence and proposes safe actions."
        ),
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [
            BATCH_MEDIA_TYPE,
        ],
        "defaultOutputModes": [
            PROPOSALS_MEDIA_TYPE,
            RECEIPTS_MEDIA_TYPE,
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Evaluates invoice evidence and proposes actions."
                ),
                "tags": [
                    "invoices",
                    "accounting",
                    "evidence",
                ],
            }
        ],
    }


# --------------------------------------------------
# Authentication and request checks
# --------------------------------------------------

def check_request(
    authorization: str | None,
    a2a_version: str | None,
    content_type: str | None,
):
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    if authorization != f"Bearer {A2A_TOKEN}":
        raise HTTPException(
            status_code=403,
            detail="Authentication failed",
        )

    if a2a_version != "1.0":
        raise HTTPException(
            status_code=400,
            detail="A2A-Version must be 1.0",
        )

    if (
        content_type is None
        or content_type.split(";")[0].strip()
        != "application/a2a+json"
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid content type",
        )


# --------------------------------------------------
# Batch validation
# --------------------------------------------------

def get_invoice_batch(message):
    parts = message.get("parts", [])

    if not isinstance(parts, list):
        raise HTTPException(
            status_code=400,
            detail="Message parts must be a list",
        )

    for part in parts:
        if not isinstance(part, dict):
            continue

        if part.get("mediaType") != BATCH_MEDIA_TYPE:
            continue

        batch = part.get("data")

        if not isinstance(batch, dict):
            raise HTTPException(
                status_code=400,
                detail="Invoice batch data must be an object",
            )

        if not isinstance(batch.get("batchId"), str) or not batch["batchId"]:
            raise HTTPException(
                status_code=400,
                detail="batchId is required",
            )

        if (
            not isinstance(batch.get("policyRevision"), str)
            or not batch["policyRevision"]
        ):
            raise HTTPException(
                status_code=400,
                detail="policyRevision is required",
            )

        packages = batch.get("packages")

        if not isinstance(packages, list):
            raise HTTPException(
                status_code=400,
                detail="packages must be a list",
            )

        package_ids = []

        for package in packages:
            if not isinstance(package, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Each package must be an object",
                )

            package_id = package.get("packageId")

            if not isinstance(package_id, str) or not package_id.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Every package needs a packageId",
                )

            package_ids.append(package_id)

        if len(package_ids) != len(set(package_ids)):
            raise HTTPException(
                status_code=400,
                detail="Package IDs must be unique",
            )

        return batch

    raise HTTPException(
        status_code=400,
        detail="Invoice batch part not found",
    )


# --------------------------------------------------
# Task storage helpers
# --------------------------------------------------

def canonical_hash(message):
    # Hash the message only; configuration is intentionally excluded.
    canonical_message = json.dumps(
        message,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical_message.encode("utf-8")
    ).hexdigest()


def save_task(task, principal, message_id, message_hash):
    with db_lock:
        connection = get_connection()

        connection.execute(
            """
            INSERT INTO tasks (
                task_id,
                principal,
                context_id,
                message_id,
                message_hash,
                task_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task["id"],
                principal,
                task["contextId"],
                message_id,
                message_hash,
                json.dumps(task),
            ),
        )

        connection.commit()
        connection.close()


def find_task_by_message(principal, message_id):
    connection = get_connection()

    row = connection.execute(
        """
        SELECT message_hash, task_json
        FROM tasks
        WHERE principal = ? AND message_id = ?
        """,
        (principal, message_id),
    ).fetchone()

    connection.close()
    return row


def get_owned_task(task_id, principal):
    connection = get_connection()

    row = connection.execute(
        """
        SELECT task_json
        FROM tasks
        WHERE task_id = ? AND principal = ?
        """,
        (task_id, principal),
    ).fetchone()

    connection.close()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    return json.loads(row["task_json"])


def update_task(task):
    with db_lock:
        connection = get_connection()

        connection.execute(
            "UPDATE tasks SET task_json = ? WHERE task_id = ?",
            (
                json.dumps(task),
                task["id"],
            ),
        )

        connection.commit()
        connection.close()


# --------------------------------------------------
# Send initial message
# --------------------------------------------------

@app.post("/message:send")
async def send_message(
    body: SendMessageBody = Body(
        ...,
        media_type="application/a2a+json",
    ),
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(
        default=None,
        alias="A2A-Version",
    ),
    content_type: str | None = Header(
        default=None,
        alias="Content-Type",
    ),
):
    check_request(
        authorization,
        a2a_version,
        content_type,
    )

    message = body.message
    principal = authorization[len("Bearer "):]

    message_id = message.get("messageId")

    if not isinstance(message_id, str) or not message_id:
        raise HTTPException(
            status_code=400,
            detail="messageId is required",
        )

    if message.get("role") != "ROLE_USER":
        raise HTTPException(
            status_code=400,
            detail="Message role must be ROLE_USER",
        )

    batch = get_invoice_batch(message)
    message_hash = canonical_hash(message)

    # Replaying the same message returns its original task.
    existing = find_task_by_message(principal, message_id)

    if existing is not None:
        if existing["message_hash"] != message_hash:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "IDEMPOTENCY_CONFLICT",
                    "message": "Message ID was reused with different content",
                },
            )

        return {"task": json.loads(existing["task_json"])}

    # IMPORTANT:
    # This currently creates a task shell only.
    # Real AI proposals and their artifact must be added next.
    task = {
        "id": str(uuid.uuid4()),
        "contextId": str(uuid.uuid4()),
        "status": {
            "state": "TASK_STATE_INPUT_REQUIRED",
        },
        "history": [message],
        "artifacts": [],
    }

    try:
        save_task(
            task,
            principal,
            message_id,
            message_hash,
        )
    except sqlite3.IntegrityError:
        # Another simultaneous request may have saved this message.
        existing = find_task_by_message(principal, message_id)

        if existing is None:
            raise HTTPException(
                status_code=409,
                detail="Could not safely save task",
            )

        if existing["message_hash"] != message_hash:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "IDEMPOTENCY_CONFLICT",
                    "message": "Message ID conflict",
                },
            )

        task = json.loads(existing["task_json"])

    return {"task": task}


# --------------------------------------------------
# Read one task
# --------------------------------------------------

@app.get("/tasks/{task_id}")
def read_task(
    task_id: str,
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(
        default=None,
        alias="A2A-Version",
    ),
    content_type: str | None = Header(
        default=None,
        alias="Content-Type",
    ),
):
    check_request(
        authorization,
        a2a_version,
        content_type,
    )

    principal = authorization[len("Bearer "):]

    return get_owned_task(task_id, principal)


# --------------------------------------------------
# List the current user's tasks
# --------------------------------------------------

@app.get("/tasks")
def list_tasks(
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(
        default=None,
        alias="A2A-Version",
    ),
    content_type: str | None = Header(
        default=None,
        alias="Content-Type",
    ),
):
    check_request(
        authorization,
        a2a_version,
        content_type,
    )

    principal = authorization[len("Bearer "):]

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT task_json
        FROM tasks
        WHERE principal = ?
        ORDER BY created_at DESC
        """,
        (principal,),
    ).fetchall()

    connection.close()

    return {
        "tasks": [
            json.loads(row["task_json"])
            for row in rows
        ]
    }


# --------------------------------------------------
# Cancel a nonterminal task
# --------------------------------------------------

@app.post("/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(
        default=None,
        alias="A2A-Version",
    ),
    content_type: str | None = Header(
        default=None,
        alias="Content-Type",
    ),
):
    check_request(
        authorization,
        a2a_version,
        content_type,
    )

    principal = authorization[len("Bearer "):]

    with db_lock:
        task = get_owned_task(task_id, principal)
        state = task["status"]["state"]

        if state in (
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED",
        ):
            raise HTTPException(
                status_code=409,
                detail="Task is already terminal",
            )

        task["status"] = {
            "state": "TASK_STATE_CANCELED",
        }

        update_task(task)

    return task
