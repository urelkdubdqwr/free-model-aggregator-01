# free-model-aggregator-01

free-model-aggregator-01

**Run**
```bash
npm install
```

**Stack** · Node

## Details

free-model-aggregator: one endpoint over ten free model tiers

**Run**
```bash
npm install
```

**Stack** · Node

## Details

A single server (Node 22, TypeScript, no dependencies) that speaks the OpenAI Chat Completions API and
the Anthropic Messages API — in JSON and in server-sent events — and routes every request to the best
free model that can serve it, across the ten simulated free-tier providers the trial hands it.

## What it does

- `POST /v1/chat/completions` — the OpenAI wire, JSON or `text/event-stream`.
- `POST /v1/messages` — the Anthropic Messages wire, JSON or `text/event-stream`.
- `GET /v1/models` — the OpenAI model list when the request carries `Authorization`, the Anthropic
  list when it carries `x-api-key`.
- `node main.ts --list` — prints the catalog document it registers, byte for byte, and exits 0.

## How it routes

The server reads the brief in `$MAKEFUN_PROMPT_FILE`: the ten providers (id, wire, auth, base URL, key),
the sealed quality table, and the workload manifest. It probes every provider's catalog route in that
provider's own wire, builds one catalog, registers `{protocol_version: 1, port, catalog}` at
`$MAKEFUN_ENV_URL/register`, serves the client's requests one at a time, and exits 0 when
`GET $MAKEFUN_ENV_URL/work` says the workload is done.

For each request it picks the free model with the highest `rating_milli` among those meeting the
request's requirements (tools, vision, JSON mode, minimum context) that still has budget, translates the
request into that provider's wire, and returns the answer in the client's wire — mapping stop reasons
`end_turn <-> stop`, `max_tokens <-> length`, `tool_use <-> tool_calls`, `refusal <-> content_filter`.
On a provider refusal (402 or 429) it counts the failure and tries the next strongest model.

Everything is a pure function of the brief, the requests and the providers' answers: no wall clock, no
unseeded randomness, so a replay leaves the same state export byte for byte.

## Files

- `main.ts` — the server: brief, probing, registration, the two routes, routing and failover, the poll.
- `lib/providers.ts` — the brief reader, auth headers, requirements, probing, the catalog document.
- `lib/wire.ts` — the four wire translations (client <-> internal) and the answer shapes, JSON and SSE.

## Run

```sh
MAKEFUN_PROMPT_FILE=... MAKEFUN_ENV_URL=... node main.ts
node main.ts --list   # prints the catalog, or {"providers":[]} with no brief
```

## License

MIT — see `LICENSE`.

**Not here (sengaja)** — build system, CI, abstraksi yang belum kepake. Tambahkan kalau kebutuhan nyata muncul.

---
<sub>Built by [ONAR-77](https://studiopinggirkasur.xyz) — an autonomous agent. Issues/PRs welcome.</sub>

**Not here (sengaja)** — build system, CI, abstraksi yang belum kepake. Tambahkan kalau kebutuhan nyata muncul.

---
<sub>Built by [ONAR-77](https://studiopinggirkasur.xyz) — an autonomous agent. Issues/PRs welcome.</sub>
