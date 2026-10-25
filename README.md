# Invoice Action Agent

FastAPI implementation of the A2A 1.0 invoice proposal and receipt API.

## Run locally

Python 3.12+ is required. Set `GROQ_API_KEY` for Groq-based invoice
understanding, then start the API:

```sh
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The agent card is at `/.well-known/agent-card.json`. For local calls, use
`Authorization: Bearer <any non-empty token>`, `A2A-Version: 1.0`, and
`Content-Type: application/a2a+json`. Each distinct bearer value has its own
task namespace.

## Deploy publicly

For a managed HTTPS deploy, create a Render Blueprint from this repository and
provide `GROQ_API_KEY` when prompted. The included `render.yaml` provisions a
Starter web service and persistent disk. Render supplies the HTTPS URL, which
the card advertises automatically.

Build and run the included Dockerfile on a container host that provides public
HTTPS and a persistent mounted volume. Mount the volume at `/data`; SQLite task
and decision state is stored at `/data/invoice_agent.db`. Configure:

| Variable | Value |
| --- | --- |
| `GROQ_API_KEY` | Your provider key, stored as a host secret |
| `DATABASE_PATH` | `/data/invoice_agent.db` |
| `AGENT_BASE_URL` | The exact public HTTPS base URL, with no trailing slash or subpath |
| `GROQ_MODEL` | Optional; defaults to `openai/gpt-oss-20b` |

The public card advertises `AGENT_BASE_URL` (or Render's `RENDER_EXTERNAL_URL`)
as its HTTP+JSON interface URL. After deploy, verify the card at
`https://<your-host>/.well-known/agent-card.json` and submit
`https://<your-host>` as the A2A base URL. Do not place bearer tokens or model
keys in the base URL.

The host must retain the mounted volume across restarts and deploys. An
ephemeral filesystem is unsuitable because it loses idempotency and receipt
state. Configure the host's health check as `/health` if it supports one.
