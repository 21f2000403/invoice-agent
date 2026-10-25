
import os
import json
import uuid
import hashlib
import sqlite3
import threading
import re
import logging
import httpx

from typing import Any
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

groq_client = None
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


class A2AJSONResponse(JSONResponse):
    media_type = "application/a2a+json"

app = FastAPI(
    title="A2A Invoice Agent",
    description="An agent for evidence-based invoice decisions",
    version="1.0.0",
    default_response_class=A2AJSONResponse,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

A2A_TOKEN = os.getenv("A2A_TOKEN", "local-test-token")
DATABASE = os.getenv("DATABASE_PATH", "invoice_agent.db")

BATCH_MEDIA_TYPE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-receipts+json"
RESULTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-results+json"

db_lock = threading.RLock()
logger = logging.getLogger("invoice_agent")
request_locks_guard = threading.Lock()
request_locks = {}


def idempotency_lock(principal, message_id):
    key = (principal, message_id)
    with request_locks_guard:
        return request_locks.setdefault(key, threading.RLock())


class SendMessageBody(BaseModel):
    message: dict[str, Any]
    configuration: dict[str, Any] = {}


# --------------------------------------------------
# Database
# --------------------------------------------------

def get_connection():
    database_directory = os.path.dirname(os.path.abspath(DATABASE))
    os.makedirs(database_directory, exist_ok=True)
    connection = sqlite3.connect(DATABASE, timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with db_lock:
        connection = get_connection()
        try:
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

            connection.execute("""
                CREATE TABLE IF NOT EXISTS continuations (
                    principal TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    message_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    PRIMARY KEY (principal, message_id)
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS decision_cache (
                    cache_key TEXT PRIMARY KEY,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            connection.commit()
        finally:
            connection.close()


@app.on_event("startup")
def startup_event():
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


@app.get("/.well-known/agent-card.json", response_class=JSONResponse)
def agent_card():
    base_url = (
        os.getenv("AGENT_BASE_URL")
        or os.getenv("RENDER_EXTERNAL_URL")
        or "http://127.0.0.1:8000"
    ).rstrip("/")

    return {
        "name": "Invoice Action Agent",
        "description": "Analyzes invoice evidence and proposes safe actions.",
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [BATCH_MEDIA_TYPE],
        "defaultOutputModes": [
            PROPOSALS_MEDIA_TYPE,
            RECEIPTS_MEDIA_TYPE,
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": "Evaluates invoice evidence and proposes actions.",
                "tags": ["invoices", "accounting", "evidence"],
            }
        ],
    }


# --------------------------------------------------
# Authentication
# --------------------------------------------------

def check_request(
    authorization: str | None,
    a2a_version: str | None,
    content_type: str | None,
):
    if authorization is None:
        raise HTTPException(401, "Authentication required")

    if not authorization.startswith("Bearer ") or not authorization[7:].strip():
        raise HTTPException(403, "Authentication failed")

    if a2a_version != "1.0":
        raise HTTPException(400, "A2A-Version must be 1.0")

    if (
        content_type is None
        or content_type.split(";")[0].strip() != "application/a2a+json"
    ):
        raise HTTPException(400, "Invalid content type")


# --------------------------------------------------
# Batch validation
# --------------------------------------------------

def get_invoice_batch(message):
    parts = message.get("parts", [])

    if not isinstance(parts, list):
        raise HTTPException(400, "Message parts must be a list")

    for part in parts:
        if not isinstance(part, dict):
            continue

        if part.get("mediaType") != BATCH_MEDIA_TYPE:
            continue

        batch = part.get("data")

        if not isinstance(batch, dict):
            raise HTTPException(400, "Invoice batch data must be an object")

        if not isinstance(batch.get("batchId"), str) or not batch["batchId"]:
            raise HTTPException(400, "batchId is required")

        if (
            not isinstance(batch.get("policyRevision"), str)
            or not batch["policyRevision"]
        ):
            raise HTTPException(400, "policyRevision is required")

        packages = batch.get("packages")

        if not isinstance(packages, list):
            raise HTTPException(400, "packages must be a list")

        package_ids = []

        for package in packages:
            if not isinstance(package, dict):
                raise HTTPException(400, "Each package must be an object")

            package_id = package.get("packageId")

            if not isinstance(package_id, str) or not package_id.strip():
                raise HTTPException(400, "Every package needs a packageId")

            package_ids.append(package_id)

        if len(package_ids) != len(set(package_ids)):
            raise HTTPException(400, "Package IDs must be unique")

        return batch

    raise HTTPException(400, "Invoice batch part not found")


# --------------------------------------------------
# Hashing and decision cache
# --------------------------------------------------

def canonical_hash(value):
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decision_cache_key(package, policy_revision):
    # Package IDs identify individual packages, not their decision content.
    content = {
        key: value
        for key, value in package.items()
        if key != "packageId"
    }

    return canonical_hash({
        "policyRevision": policy_revision,
        "package": content,
    })


def get_cached_decision(cache_key):
    with db_lock:
        connection = get_connection()
        try:
            row = connection.execute(
                "SELECT decision_json FROM decision_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        finally:
            connection.close()

    if row is None:
        return None

    try:
        decision = json.loads(row["decision_json"])
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(decision, dict):
        return None

    return decision


def save_cached_decision(cache_key, decision):
    # Cache semantic decisions only. Never cache packageId or actionId.
    decision_json = json.dumps(
        decision,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    with db_lock:
        connection = get_connection()
        try:
            connection.execute(
                """
                INSERT INTO decision_cache (cache_key, decision_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key)
                DO UPDATE SET decision_json = excluded.decision_json
                """,
                (cache_key, decision_json),
            )
            connection.commit()
        finally:
            connection.close()


# --------------------------------------------------
# Task storage helpers
# --------------------------------------------------

def save_task(task, principal, message_id, message_hash):
    with db_lock:
        connection = get_connection()
        try:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, principal, context_id, message_id,
                    message_hash, task_json
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
        finally:
            connection.close()


def find_task_by_message(principal, message_id):
    connection = get_connection()
    try:
        return connection.execute(
            """
            SELECT message_hash, task_json
            FROM tasks
            WHERE principal = ? AND message_id = ?
            """,
            (principal, message_id),
        ).fetchone()
    finally:
        connection.close()


def get_owned_task(task_id, principal):
    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT task_json
            FROM tasks
            WHERE task_id = ? AND principal = ?
            """,
            (task_id, principal),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise HTTPException(404, "Task not found")

    return json.loads(row["task_json"])


def update_task(task):
    with db_lock:
        connection = get_connection()
        try:
            connection.execute(
                "UPDATE tasks SET task_json = ? WHERE task_id = ?",
                (json.dumps(task), task["id"]),
            )
            connection.commit()
        finally:
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
    a2a_version: str | None = Header(default=None, alias="A2A-Version"),
    content_type: str | None = Header(default=None, alias="Content-Type"),
):
    check_request(authorization, a2a_version, content_type)

    message = body.message
    principal = authorization[len("Bearer "):]
    message_id = message.get("messageId")

    if not isinstance(message_id, str) or not message_id:
        raise HTTPException(400, "messageId is required")

    if message.get("role") != "ROLE_USER":
        raise HTTPException(400, "Message role must be ROLE_USER")

    parts = message.get("parts", [])
    is_results_message = (
        isinstance(parts, list)
        and any(
            isinstance(part, dict)
            and part.get("mediaType") == RESULTS_MEDIA_TYPE
            for part in parts
        )
    )

    if is_results_message:
        task = process_results_message(message, principal)
        return {"task": task}

    batch = get_invoice_batch(message)
    message_hash = canonical_hash(message)

    # Serialize same-key requests across decision generation and persistence.
    with idempotency_lock(principal, message_id):
        return create_or_replay_task(message, batch, principal, message_id, message_hash)


def create_or_replay_task(message, batch, principal, message_id, message_hash):

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

    proposals = generate_proposals(batch)

    task = {
        "id": str(uuid.uuid4()),
        "contextId": str(uuid.uuid4()),
        "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
        "history": [message],
        "artifacts": [
            {
                "artifactId": str(uuid.uuid4()),
                "name": "Invoice Action Proposals",
                "parts": [
                    {
                        "mediaType": PROPOSALS_MEDIA_TYPE,
                        "data": {
                            "batchId": batch["batchId"],
                            "proposals": proposals,
                        },
                    }
                ],
            }
        ],
    }

    try:
        save_task(task, principal, message_id, message_hash)
    except sqlite3.IntegrityError:
        existing = find_task_by_message(principal, message_id)

        if existing is None:
            raise HTTPException(409, "Could not safely save task")

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
# Read task
# --------------------------------------------------

@app.get("/tasks/{task_id}")
def read_task(
    task_id: str,
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(default=None, alias="A2A-Version"),
    content_type: str | None = Header(default=None, alias="Content-Type"),
):
    check_request(authorization, a2a_version, content_type)
    principal = authorization[len("Bearer "):]
    return get_owned_task(task_id, principal)


# --------------------------------------------------
# List tasks
# --------------------------------------------------

@app.get("/tasks")
def list_tasks(
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(default=None, alias="A2A-Version"),
    content_type: str | None = Header(default=None, alias="Content-Type"),
):
    check_request(authorization, a2a_version, content_type)
    principal = authorization[len("Bearer "):]

    connection = get_connection()
    try:
        rows = connection.execute(
            """
            SELECT task_json FROM tasks
            WHERE principal = ?
            ORDER BY created_at DESC
            """,
            (principal,),
        ).fetchall()
    finally:
        connection.close()

    return {"tasks": [json.loads(row["task_json"]) for row in rows]}


# --------------------------------------------------
# Cancel task
# --------------------------------------------------

@app.post("/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(default=None, alias="A2A-Version"),
    content_type: str | None = Header(default=None, alias="Content-Type"),
):
    check_request(authorization, a2a_version, content_type)
    principal = authorization[len("Bearer "):]

    with db_lock:
        task = get_owned_task(task_id, principal)
        state = task["status"]["state"]

        if state in (
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED",
        ):
            raise HTTPException(409, "Task is already terminal")

        task["status"] = {"state": "TASK_STATE_CANCELED"}
        update_task(task)

    return task


# --------------------------------------------------
# Groq invoice proposal generation and caching
# --------------------------------------------------

ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}


def extract_references(value):
    references = set()

    if isinstance(value, dict):
        for item in value.values():
            references.update(extract_references(item))

    elif isinstance(value, list):
        for item in value:
            references.update(extract_references(item))

    elif isinstance(value, str):
        references.update(re.findall(r"\[[^\[\]\n]{1,160}\]", value))

    return references


def contains_value(value, target):
    if isinstance(value, dict):
        return any(contains_value(item, target) for item in value.values())

    if isinstance(value, list):
        return any(contains_value(item, target) for item in value)

    return type(value) is type(target) and value == target


def validate_decision(decision, package, available_refs):
    if not isinstance(decision, dict):
        raise HTTPException(502, "Invalid cached or generated decision")

    action = decision.get("action")
    facts = decision.get("facts")
    refs = decision.get("evidenceRefs")
    rationale = decision.get("rationale")

    if action not in ALLOWED_ACTIONS:
        raise HTTPException(502, "Proposal action is not allowed")

    if not isinstance(facts, dict):
        raise HTTPException(502, "Proposal facts are invalid")

    vendor = facts.get("vendorName")
    invoice_number = facts.get("invoiceNumber")
    amount = facts.get("amountMinor")
    currency = facts.get("currency")

    if (
        not isinstance(vendor, str)
        or not vendor.strip()
        or not isinstance(invoice_number, str)
        or not invoice_number.strip()
        or isinstance(amount, bool)
        or not isinstance(amount, int)
        or amount < 0
        or not isinstance(currency, str)
        or not currency.strip()
    ):
        raise HTTPException(502, "Required invoice facts are invalid")

    for fact in (vendor, invoice_number, amount, currency):
        if not contains_value(package, fact):
            raise HTTPException(
                502,
                "Proposal contains an unverified invoice fact",
            )

    if (
        not isinstance(refs, list)
        or len(refs) != 3
        or any(not isinstance(ref, str) for ref in refs)
        or len(set(refs)) != 3
    ):
        raise HTTPException(
            502,
            "Each proposal needs three unique evidence references",
        )

    if not set(refs).issubset(set(available_refs)):
        raise HTTPException(
            502,
            "Proposal contains an unverified evidence reference",
        )

    if (
        not isinstance(rationale, str)
        or not 60 <= len(rationale) <= 1500
        or sum(ref in rationale for ref in refs) < 2
    ):
        raise HTTPException(502, "Proposal rationale is invalid")

    return {
        "action": action,
        "facts": {
            "vendorName": vendor,
            "invoiceNumber": invoice_number,
            "amountMinor": amount,
            "currency": currency,
        },
        "evidenceRefs": refs,
        "rationale": rationale,
    }


def generate_proposals(batch):
    packages = batch["packages"]

    if not packages:
        return []

    policy_revision = batch["policyRevision"]
    evidence_by_id = {}
    key_by_id = {}
    package_by_id = {}
    decisions_by_key = {}
    uncached_by_key = {}

    # Determine cache hits and collect all cache misses.
    # Identical package contents share a cache key even if IDs differ.
    for package in packages:
        package_id = package["packageId"]
        refs = sorted(extract_references(package))

        if len(refs) < 3:
            raise HTTPException(
                422,
                f"Package {package_id} has fewer than three "
                "bracketed evidence references",
            )

        cache_key = decision_cache_key(package, policy_revision)

        evidence_by_id[package_id] = refs
        key_by_id[package_id] = cache_key
        package_by_id[package_id] = package

        if cache_key in decisions_by_key:
            continue

        cached = get_cached_decision(cache_key)

        if cached is not None:
            try:
                decisions_by_key[cache_key] = validate_decision(
                    cached, package, refs
                )
                continue
            except HTTPException:
                # An invalid cache entry is treated as a cache miss.
                pass

        if cache_key not in uncached_by_key:
            uncached_by_key[cache_key] = package

    # Send every unique cache miss in ONE Groq request.
    if uncached_by_key:
        prompt_data = []

        for cache_key, package in uncached_by_key.items():
            prompt_data.append({
                "package": package,
                "availableEvidenceRefs": sorted(
                    extract_references(package)
                ),
            })

        prompt = f"""
You are an evidence-based invoice decision assistant.

Analyze each package independently. Return one proposal for each
package supplied in this request.

Allowed actions:
- settle_invoice
- request_approval
- hold_invoice
- reject_duplicate
- open_exception

Return a JSON object with this structure:
{{
  "proposals": [
    {{
      "packageId": "exact package ID",
      "action": "one allowed action",
      "facts": {{
        "vendorName": "exact vendor name",
        "invoiceNumber": "exact invoice number",
        "amountMinor": 12500,
        "currency": "USD"
      }},
      "evidenceRefs": [
        "[exact reference 1]",
        "[exact reference 2]",
        "[exact reference 3]"
      ],
      "rationale": "Explain the decision using at least two references."
    }}
  ]
}}

Rules:
1. Return exactly one proposal per supplied package, using each packageId once.
2. Evaluate each package independently using its own evidence.
3. A duplicate must be the same invoice or payment obligation.
   A different invoice from the same vendor is not automatically a duplicate.
4. Old examples, quoted messages, and unrelated records are not proof
   about the current invoice.
5. Interpret negation, corrections, dates, invoice numbers, amounts,
   and vendor identities in context.
6. If evidence conflicts or facts are uncertain, prefer hold_invoice
   or open_exception.
7. Use request_approval only when evidence establishes approval is required.
   Do not invent a policy threshold.
8. Use settle_invoice only when evidence supports payment and no unresolved
   duplicate, approval, or material conflict remains.
9. Use reject_duplicate only when evidence supports a genuine duplicate.
10. Copy facts from the current invoice evidence. Never invent missing facts.
11. Select exactly three distinct, relevant references from the package's
    availableEvidenceRefs.
12. Explain the decision in 60-1500 characters and cite at least two
    selected references exactly.
13. Return valid JSON only.

Input:
{json.dumps(prompt_data, ensure_ascii=False)}
"""

        try:
            provider = os.getenv(
                "AI_PROVIDER",
                "gemini" if os.getenv("GOOGLE_API_KEY") else "groq",
            ).lower()

            if provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise RuntimeError("GOOGLE_API_KEY is not configured")
                model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
                response = httpx.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key},
                    json={
                        "contents": [{
                            "role": "user",
                            "parts": [{"text": prompt}],
                        }],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0,
                        },
                    },
                    timeout=40.0,
                )
                response.raise_for_status()
                response_data = response.json()
                result_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(result_text)
            elif provider == "groq":
                global groq_client
                if groq_client is None:
                    api_key = os.getenv("GROQ_API_KEY")
                    if not api_key:
                        raise RuntimeError("GROQ_API_KEY is not configured")
                    groq_client = Groq(api_key=api_key)

                response = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return valid JSON only. Never invent invoice "
                                "facts or evidence references."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                result = json.loads(response.choices[0].message.content)
            else:
                raise RuntimeError("AI_PROVIDER must be gemini or groq")

        except Exception as exc:
            logger.exception(
                "Invoice proposal provider call failed (%s)",
                type(exc).__name__,
            )
            raise HTTPException(
                502,
                "Invoice proposal generation failed",
            ) from exc

        proposals = result.get("proposals")

        if (
            not isinstance(proposals, list)
            or len(proposals) != len(uncached_by_key)
        ):
            raise HTTPException(
                502,
                "AI returned an invalid number of proposals",
            )

        # The model responds using the representative package IDs.
        representative_id_to_key = {
            package["packageId"]: cache_key
            for cache_key, package in uncached_by_key.items()
        }

        seen_keys = set()
        fresh_decisions = {}

        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise HTTPException(502, "Invalid proposal object")

            package_id = proposal.get("packageId")
            cache_key = representative_id_to_key.get(package_id)

            if cache_key is None or cache_key in seen_keys:
                raise HTTPException(
                    502,
                    "Proposal package IDs are invalid",
                )

            package = uncached_by_key[cache_key]
            refs = sorted(extract_references(package))

            # Only semantic fields are stored in the cache.
            decision = validate_decision(proposal, package, refs)
            fresh_decisions[cache_key] = decision
            seen_keys.add(cache_key)

        if len(fresh_decisions) != len(uncached_by_key):
            raise HTTPException(
                502,
                "AI did not return every required decision",
            )

        # Cache decisions only after all generated decisions validate.
        for cache_key, decision in fresh_decisions.items():
            save_cached_decision(cache_key, decision)
            decisions_by_key[cache_key] = decision

    # Build a proposal for every original package.
    # Always create a new actionId, including for cache hits.
    validated_proposals = []

    for package in packages:
        package_id = package["packageId"]
        cache_key = key_by_id[package_id]
        decision = decisions_by_key.get(cache_key)

        if decision is None:
            raise HTTPException(502, "Missing invoice decision")

        # Validate against this particular package as well.
        decision = validate_decision(
            decision,
            package,
            evidence_by_id[package_id],
        )

        validated_proposals.append({
            "packageId": package_id,
            "actionId": str(uuid.uuid4()),
            "action": decision["action"],
            "facts": decision["facts"],
            "evidenceRefs": decision["evidenceRefs"],
            "rationale": decision["rationale"],
        })

    return validated_proposals


# --------------------------------------------------
# Process results and create receipts
# --------------------------------------------------

def process_results_message(message, principal):
    if message.get("role") != "ROLE_USER":
        raise HTTPException(400, "Message role must be ROLE_USER")
    message_id = message.get("messageId")
    task_id = message.get("taskId")
    context_id = message.get("contextId")

    if not isinstance(message_id, str) or not message_id:
        raise HTTPException(400, "messageId is required")

    if not isinstance(task_id, str) or not task_id:
        raise HTTPException(400, "taskId is required")

    if not isinstance(context_id, str) or not context_id:
        raise HTTPException(400, "contextId is required")

    message_hash = canonical_hash(message)

    with db_lock:
        connection = get_connection()

        previous = connection.execute(
            """
            SELECT message_hash, response_json
            FROM continuations
            WHERE principal = ? AND message_id = ?
            """,
            (principal, message_id),
        ).fetchone()

        if previous is not None:
            connection.close()

            if previous["message_hash"] != message_hash:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "IDEMPOTENCY_CONFLICT",
                        "message": "Message ID was reused with different content",
                    },
                )

            return json.loads(previous["response_json"])

        row = connection.execute(
            """
            SELECT task_json
            FROM tasks
            WHERE task_id = ? AND principal = ?
            """,
            (task_id, principal),
        ).fetchone()

        if row is None:
            connection.close()
            raise HTTPException(404, "Task not found")

        task = json.loads(row["task_json"])

        if task["contextId"] != context_id:
            connection.close()
            raise HTTPException(400, "Context ID does not match")

        if task["status"]["state"] in (
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED",
        ):
            connection.close()
            raise HTTPException(409, "Task is already terminal")

        results_data = None

        for part in message.get("parts", []):
            if (
                isinstance(part, dict)
                and part.get("mediaType") == RESULTS_MEDIA_TYPE
            ):
                results_data = part.get("data")
                break

        if not isinstance(results_data, dict):
            connection.close()
            raise HTTPException(400, "Results data must be an object")

        batch_id = results_data.get("batchId")
        results = results_data.get("results")

        proposal_artifact = next(
            (
                artifact
                for artifact in task.get("artifacts", [])
                if any(
                    isinstance(part, dict)
                    and part.get("mediaType") == PROPOSALS_MEDIA_TYPE
                    and isinstance(part.get("data"), dict)
                    and part["data"].get("batchId") == batch_id
                    for part in artifact.get("parts", [])
                )
            ),
            None,
        )

        if proposal_artifact is None:
            connection.close()
            raise HTTPException(400, "Batch ID does not match the task")

        proposals = next(
            part["data"]["proposals"]
            for part in proposal_artifact["parts"]
            if part.get("mediaType") == PROPOSALS_MEDIA_TYPE
        )

        if not isinstance(results, list) or not results:
            connection.close()
            raise HTTPException(400, "Results must be a non-empty list")

        submitted_ids = [
            item.get("packageId") for item in results if isinstance(item, dict)
        ]
        if len(submitted_ids) != len(proposals) or set(submitted_ids) != set(
            proposal["packageId"] for proposal in proposals
        ) or len(set(submitted_ids)) != len(submitted_ids):
            connection.close()
            raise HTTPException(400, "Results must match every proposal exactly once")

        proposals_by_id = {
            proposal["packageId"]: proposal
            for proposal in proposals
        }

        previous_results = {}

        for old_message in task.get("history", []):
            for part in old_message.get("parts", []):
                if (
                    isinstance(part, dict)
                    and part.get("mediaType") == RESULTS_MEDIA_TYPE
                ):
                    old_data = part.get("data", {})
                    if old_data.get("batchId") == batch_id:
                        for old_result in old_data.get("results", []):
                            previous_results[old_result.get("packageId")] = (
                                old_result
                            )

        accepted_executions = []
        current_result_ids = set()

        for result in results:
            if not isinstance(result, dict):
                connection.close()
                raise HTTPException(400, "Each result must be an object")

            package_id = result.get("packageId")
            action_id = result.get("actionId")
            action = result.get("action")
            outcome = result.get("outcome")
            nonce = result.get("receiptNonce")

            if package_id not in proposals_by_id:
                connection.close()
                raise HTTPException(400, "Unknown package ID")

            if package_id in previous_results or package_id in current_result_ids:
                connection.close()
                raise HTTPException(
                    409,
                    "Package result was already processed or repeated",
                )

            proposal = proposals_by_id[package_id]

            if (
                action_id != proposal["actionId"]
                or action != proposal["action"]
            ):
                connection.close()
                raise HTTPException(400, "Action does not match the proposal")

            if outcome not in ("ACCEPTED", "REJECTED"):
                connection.close()
                raise HTTPException(400, "Invalid result outcome")

            if not isinstance(nonce, str) or not nonce.strip():
                connection.close()
                raise HTTPException(400, "receiptNonce is required")

            if outcome == "ACCEPTED":
                accepted_executions.append({
                    "packageId": package_id,
                    "actionId": action_id,
                    "action": action,
                    "receiptNonce": nonce,
                    "facts": proposal["facts"],
                    "evidenceRefs": proposal["evidenceRefs"],
                })

            current_result_ids.add(package_id)
            previous_results[package_id] = result

        task["history"].append(message)

        receipt_artifact = next(
            (
                artifact
                for artifact in task.get("artifacts", [])
                if any(
                    isinstance(part, dict)
                    and part.get("mediaType") == RECEIPTS_MEDIA_TYPE
                    for part in artifact.get("parts", [])
                )
            ),
            None,
        )

        if receipt_artifact is None:
            receipt_artifact = {
                "artifactId": str(uuid.uuid4()),
                "name": "Invoice Action Receipts",
                "parts": [
                    {
                        "mediaType": RECEIPTS_MEDIA_TYPE,
                        "data": {
                            "batchId": batch_id,
                            "executions": [],
                        },
                    }
                ],
            }
            task["artifacts"].append(receipt_artifact)

        receipt_data = next(
            part["data"]
            for part in receipt_artifact["parts"]
            if part.get("mediaType") == RECEIPTS_MEDIA_TYPE
        )

        if receipt_data.get("batchId") != batch_id:
            connection.close()
            raise HTTPException(400, "Receipt batch ID does not match")

        receipt_data["executions"].extend(accepted_executions)

        if len(previous_results) == len(proposals_by_id):
            task["status"] = {"state": "TASK_STATE_COMPLETED"}
        else:
            task["status"] = {"state": "TASK_STATE_INPUT_REQUIRED"}

        response_json = json.dumps(task)

        connection.execute(
            """
            INSERT INTO continuations (
                principal, message_id, message_hash, task_id, response_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                principal,
                message_id,
                message_hash,
                task_id,
                response_json,
            ),
        )

        connection.execute(
            """
            UPDATE tasks
            SET task_json = ?
            WHERE task_id = ? AND principal = ?
            """,
            (response_json, task_id, principal),
        )

        connection.commit()
        connection.close()

        return task
