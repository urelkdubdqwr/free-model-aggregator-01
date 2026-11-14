// The client wires (OpenAI Chat Completions, Anthropic Messages) and all three upstream wires
// (openai, anthropic, gemini) this entry speaks.
//
// The entry keeps ONE internal request shape: the OpenAI Chat Completions body. A client request on
// either wire is turned into that shape, the shape is turned into the chosen provider's wire, the
// provider's JSON answer is read back into one internal answer (text, usage, finish reason), and the
// internal answer is rendered in the wire the client used, streamed or not.
//
// Faithfulness is what scores: the simulated provider echoes back everything it parsed from the
// request it received (a sha of the system prompt and of the user text, the tool names, the output
// cap, the stop sequences, the image bytes and whether JSON mode was carried). Every translation here
// therefore preserves those fields exactly, so the echo the provider produces equals the echo the
// client expects. A JSON-mode request can only be carried by the openai and gemini wires: the
// anthropic wire has no field for it, so such a request is never routed to an anthropic provider.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so a replay sees
// the same bytes. The model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** One provider answer, read back into the internal shape. `finish` is the provider's own stop word. */
export interface InternalAnswer {
  text: string;
  usage: { input: number; output: number };
  finish: string;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** `stop` as the provider wires write it: a string or an array of strings, always an array here. */
function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** A data URL split into its media type and its base64 payload; null when it is not one. */
function dataUrlOf(url: unknown): { media: string; data: string } | null {
  if (typeof url !== 'string') return null;
  const m = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const inline = dataUrlOf(part.image_url.url);
    if (inline !== null) return { type: 'image', source: { type: 'base64', media_type: inline.media, data: inline.data } };
    if (typeof part.image_url.url === 'string') return { type: 'image', source: { type: 'url', url: part.image_url.url } };
  }
  return null;
}

/** One OpenAI tool definition as an Anthropic tool definition. */
function openAIToolToAnthropic(tool: Json): Json | null {
  if (!isRecord(tool.function) || typeof tool.function.name !== 'string') return null;
  return {
    name: tool.function.name,
    description: typeof tool.function.description === 'string' ? tool.function.description : '',
    input_schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
  };
}

/**
 * An internal (Chat Completions) request as an Anthropic Messages request. The system prompt becomes
 * `system`, image parts become image blocks, tools become `{name, description, input_schema}`, the
 * output cap becomes `max_tokens` and the stop sequences become `stop_sequences`. Everything the
 * provider's parser reads is preserved, so its echo equals the client's expectation.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToAnthropic(part);
      if (block !== null) blocks.push(block);
    }
    messages.push({ role, content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens;
  if (typeof maxTokens === 'number') out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((t) => (isRecord(t) ? [openAIToolToAnthropic(t)].filter((x): x is Json => x !== null) : []));
    if (tools.length > 0) out.tools = tools;
  }
  return out;
}

/** One OpenAI content part as a Gemini part: text or inline image data; null for the rest. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) {
    const inline = dataUrlOf(part.image_url.url);
    if (inline !== null) return { inlineData: { mimeType: inline.media, data: inline.data } };
    if (typeof part.image_url.url === 'string') return { fileData: { fileUri: part.image_url.url } };
  }
  return null;
}

/**
 * An internal (Chat Completions) request as a Gemini generateContent request. The system prompt
 * becomes `systemInstruction`, messages become `contents`, tools become `functionDeclarations`, the
 * output cap and stop sequences become `generationConfig`, and JSON mode becomes
 * `generationConfig.responseMimeType` — the field Gemini's parser reads as JSON mode.
 */
export function openAIRequestToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role, parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const translated = openAIPartToGemini(part);
      if (translated !== null) parts.push(translated);
    }
    contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((t) =>
      isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string'
        ? [{ name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : '', parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } }]
        : [],
    );
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens;
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  const stops = stringList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Anthropic's stop_reason from a provider's own stop word. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length' || finish === 'max_tokens' || finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call' || finish === 'tool_use') return 'tool_use';
  if (finish === 'content_filter' || finish === 'refusal' || finish === 'SAFETY') return 'content_filter';
  return 'end_turn';
}

/** The stop word a client of the given wire expects, from the provider's own stop word. */
export function stopWordFor(wire: 'openai' | 'anthropic', finish: unknown): string {
  const reason = stopReasonOf(finish); // an anthropic-flavoured reason
  if (wire === 'openai') {
    if (reason === 'max_tokens') return 'length';
    if (reason === 'tool_use') return 'tool_calls';
    if (reason === 'content_filter') return 'content_filter';
    return 'stop';
  }
  return reason;
}

/** The text of an OpenAI completion's message content. */
function openAIMessageText(message: Json): string {
  if (typeof message.content === 'string') return message.content;
  return textOf(message.content);
}

/** A Gemini answer's first candidate text: its parts' text joined. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRecord(first.content) ? first.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** Read an OpenAI Chat Completions answer into the internal shape. */
export function openAIAnswerToInternal(body: Json): InternalAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(first.message) ? first.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: openAIMessageText(message),
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
    finish: typeof first.finish_reason === 'string' ? first.finish_reason : 'stop',
  };
}

/** Read an Anthropic Messages answer into the internal shape. */
export function anthropicAnswerToInternal(body: Json): InternalAnswer {
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
    finish: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** Read a Gemini generateContent answer into the internal shape. */
export function geminiAnswerToInternal(body: Json): InternalAnswer {
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return {
    text: geminiTextOf(body),
    usage: { input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) },
    finish: typeof first.finishReason === 'string' ? first.finishReason : 'STOP',
  };
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicPartToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an internal (Chat Completions) request: the system prompt becomes
 * a system message, image blocks become image_url data URLs, tool_use blocks become tool_calls,
 * tool_result blocks become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicRequestToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) } });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicPartToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) =>
      !isRecord(raw) || typeof raw.name !== 'string'
        ? []
        : [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }],
    );
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The internal answer as the OpenAI JSON body the client expects. */
export function openAIAnswerBody(answer: InternalAnswer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stopWordFor('openai', answer.finish), logprobs: null }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  };
}

/** The internal answer as the OpenAI SSE frames the client expects: deltas, a finish, a usage chunk, [DONE]. */
export function openAIAnswerFrames(answer: InternalAnswer, model: string): string[] {
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  return [
    sseEvent(null, chunk({ role: 'assistant', content: '' }, null)),
    sseEvent(null, chunk({ content: answer.text }, null)),
    sseEvent(null, chunk({}, stopWordFor('openai', answer.finish))),
    sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output } }),
    'data: [DONE]\n\n',
  ];
}

/** The internal answer as the Anthropic JSON message the client expects. */
export function anthropicAnswerBody(answer: InternalAnswer, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopWordFor('anthropic', answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** The internal answer as the Anthropic SSE event sequence the client expects. */
export function anthropicAnswerFrames(answer: InternalAnswer, model: string): string[] {
  return [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopWordFor('anthropic', answer.finish), stop_sequence: null }, usage: { output_tokens: answer.usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ];
}
