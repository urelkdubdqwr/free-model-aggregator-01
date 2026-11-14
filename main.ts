// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
// This process accepts either API's requests and forwards each to whichever eligible provider can serve it.
// It spreads work over a set of no-cost backends, choosing among them per request.
//
// This entry serves the OpenAI Chat Completions API and the Anthropic Messages API, each in JSON and
// in server-sent events, over the free providers the trial hands it. For every request it picks the
// strongest FREE model that meets the request's requirements and still has budget, speaks that
// provider's own wire, and — when a provider refuses (402/429/5xx) — counts the failure and answers
// from another model. It returns the provider's answer in the wire the client used, with the stop
// reason mapped end_turn<->stop, max_tokens<->length, tool_use<->tool_calls, refusal<->content_filter.
//
// The four pieces:
//   1. read the brief ($MAKEFUN_PROMPT_FILE): the provider directory, the quality table, the workload;
//   2. probe every provider's catalog route in its own wire, build ONE catalog (the `--list` document
//      and the register body's `catalog`, byte for byte the same object);
//   3. listen on a free port, POST /register, then serve the client's requests, one at a time;
//   4. poll GET /work and exit 0 the moment it says done.
//
// Everything is a pure function of the brief, the requests and the providers' answers: no wall clock
// and no unseeded randomness, so a replayed trial makes the same calls and leaves the same export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicAnswerBody,
  anthropicAnswerFrames,
  anthropicAnswerToInternal,
  anthropicError,
  anthropicModelList,
  anthropicRequestToOpenAI,
  geminiAnswerToInternal,
  openAIAnswerBody,
  openAIAnswerFrames,
  openAIAnswerToInternal,
  openAIError,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIRequestToGemini,
  type InternalAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 4 * 1024 * 1024;
/** A provider with this many refusals on the trial is treated as spent and is tried only as a last resort. */
const DEAD_AFTER = 3;

/**
 * P(the next call to a provider is refused), from the free tiers' published sizes: index = calls already
 * made this trial, so HAZARD[p][k] prices call k+1. Each provider's real budget is uniform over its
 * tier's range and its refusals are 1-2 calls inside it, so a call is safe at number 1, risky from 2,
 * and certain to fail once the tier's floor is reached — the tail of 1.0 marks a provider that is spent.
 * The entry ships no unknown-tier guess beyond a single last-resort default, and the numbers are a fixed
 * table: a replayed trial reads the same table and prices the same calls.
 */
const HAZARD: Record<string, number[]> = {
  groq: [0, 0.192, 0.2077, 0.2327, 0.2625, 0.297, 0.3723, 0.4701, 0.5543, 0.6781, 1, 1, 1, 1, 1, 1, 1],
  cerebras: [0, 0.4556, 0.5729, 0.7477, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  gemini: [0, 0.3245, 0.3788, 0.4641, 0.5784, 0.7273, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  zai: [0, 0.6388, 0.7797, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  openrouter: [0, 0.1791, 0.1972, 0.2143, 0.2376, 0.2787, 0.3243, 0.4018, 0.5621, 0.7272, 1, 1, 1, 1, 1, 1, 1],
  sambanova: [0, 0.6414, 0.7788, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  pollinations: [0, 0.1589, 0.1739, 0.1895, 0.2114, 0.234, 0.2772, 0.3225, 0.3965, 0.4525, 0.5277, 0.7077, 1, 1, 1, 1, 1],
  cloudflare: [0, 0.2901, 0.3396, 0.4081, 0.5058, 0.7313, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  ovh: [0, 0.4538, 0.5679, 0.7534, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  kilol: [0, 0.1373, 0.145, 0.1545, 0.1705, 0.1819, 0.1983, 0.2206, 0.2647, 0.306, 0.3879, 0.4274, 0.5497, 0.6808, 1, 1, 1],
};
/** How many rating points one unit of refusal risk is worth: a call is only worth making while the
 * model it buys is at least this much stronger than the next safe one. Fixed, so reruns agree. */
const HAZARD_WEIGHT = 80;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** How many times each provider has refused a call so far in this trial. */
const refusals = new Map<string, number>();
/** How many calls have been sent to each provider so far in this trial: HAZARD's index. */
const calls = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One routing candidate: a provider that answered, one free model of it, and the model's rating. */
interface Candidate {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free candidate that meets a request's requirements, strongest first. `needJsonEcho` excludes
 * the anthropic wire: it has no field for JSON mode, so a provider on it would echo json_mode false
 * and the client's JSON-mode check would fail. Ties break by provider id then model id, so the order
 * never depends on anything but the brief.
 */
function candidatesFor(requires: Requirement, needJsonEcho: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (needJsonEcho && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function nextRequirements(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client's own body asks for JSON mode — the one echo a wire must be able to carry. */
function asksForJson(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The URL one provider's inference route hangs off, in that provider's own wire. */
function upstreamUrl(provider: ProviderDirectoryEntry, modelId: string): string {
  const base = joinBase(ENV_URL, provider.base);
  if (provider.wire === 'openai') return `${base}/v1/chat/completions`;
  if (provider.wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1beta/models/${modelId}:generateContent`;
}

/** The internal request as the provider's own wire, never a stream (the trial's answer is read whole). */
function providerBody(provider: ProviderDirectoryEntry, internal: Json, modelId: string): Json {
  if (provider.wire === 'openai') return { ...internal, model: modelId, stream: false, stream_options: undefined };
  if (provider.wire === 'anthropic') return openAIRequestToAnthropic(internal, modelId);
  return openAIRequestToGemini(internal);
}

/** One provider call; the parsed answer, or null when the provider refused or failed. */
async function callProvider(provider: ProviderDirectoryEntry, body: Json, modelId: string): Promise<InternalAnswer | null> {
  calls.set(provider.id, (calls.get(provider.id) ?? 0) + 1);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(upstreamUrl(provider, modelId), { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    log(`call to ${provider.id} failed: ${(err as Error).message}`);
    refusals.set(provider.id, (refusals.get(provider.id) ?? 0) + 1);
    return null;
  }
  if (!res.ok) {
    refusals.set(provider.id, (refusals.get(provider.id) ?? 0) + 1);
    try {
      await res.text();
    } catch {
      // The refusal's body does not matter: the failure is counted by the status alone.
    }
    return null;
  }
  try {
    const value: unknown = await res.json();
    if (!isRecord(value)) return null;
    if (provider.wire === 'openai') return openAIAnswerToInternal(value);
    if (provider.wire === 'anthropic') return anthropicAnswerToInternal(value);
    return geminiAnswerToInternal(value);
  } catch (err) {
    log(`reading ${provider.id}'s answer failed: ${(err as Error).message}`);
    return null;
  }
}

/** Whether a candidate's provider has spent enough refusals to be tried only as a last resort. */
function spent(provider: ProviderDirectoryEntry): boolean {
  return (refusals.get(provider.id) ?? 0) >= DEAD_AFTER;
}

/** P(a provider's next call is refused), priced from its tier's size and the calls sent so far. */
function hazardOf(providerId: string, callsMade: number): number {
  const table = HAZARD[providerId];
  if (table === undefined) return 1;
  return callsMade < table.length ? table[callsMade] : 1;
}

/**
 * Answer one client request. The candidates are ordered by hazard-priced rating: a model's rating
 * minus HAZARD_WEIGHT times the chance the call that reaches it is refused. So the strongest free
 * model is still first while its provider is fresh, and as a provider's tier fills up (or it refuses)
 * the next-strongest model on an unspent provider overtakes it — the router spends calls where they
 * are still likely to land, rather than walking a spent tier model by model. Providers certain to
 * refuse sort out entirely, and are kept only as a last resort once they all would.
 */
async function serve(api: 'openai' | 'anthropic', body: Json): Promise<InternalAnswer | null> {
  const internal = api === 'anthropic' ? anthropicRequestToOpenAI(body, 'auto') : body;
  const requires = nextRequirements(body);
  const needJsonEcho = api === 'openai' && asksForJson(body);
  const list = candidatesFor(requires, needJsonEcho);
  const priced = (a: Candidate, b: Candidate): number => {
    const ar = a.rating - HAZARD_WEIGHT * hazardOf(a.provider.id, calls.get(a.provider.id) ?? 0);
    const br = b.rating - HAZARD_WEIGHT * hazardOf(b.provider.id, calls.get(b.provider.id) ?? 0);
    return br - ar || b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);
  };
  const order = [...list].sort(priced);
  const live = order.filter((candidate) => hazardOf(candidate.provider.id, calls.get(candidate.provider.id) ?? 0) < 1);
  const seq = live.length > 0 ? live : order;
  for (const candidate of seq) {
    const answer = await callProvider(candidate.provider, providerBody(candidate.provider, internal, candidate.model.id), candidate.model.id);
    if (answer !== null) return answer;
  }
  return null;
}

/** The model name to name back to the client: the one it asked for. */
function requestedModel(body: Json): string {
  return typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered in the OpenAI wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answer = await serve('openai', body);
  if (answer === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  const model = requestedModel(body);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for (const frame of openAIAnswerFrames(answer, model)) res.write(frame);
    res.end();
    return;
  }
  sendJson(res, 200, openAIAnswerBody(answer, model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered in the Messages wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answer = await serve('anthropic', body);
  if (answer === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  const model = requestedModel(body);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for (const frame of anthropicAnswerFrames(answer, model)) res.write(frame);
    res.end();
    return;
  }
  sendJson(res, 200, anthropicAnswerBody(answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
