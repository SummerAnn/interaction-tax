/**
 * Minimal LLM Client for benchmark experiments.
 * Supports Anthropic, OpenAI, and OpenRouter APIs.
 */

import type { BackboneConfig, TokenUsage } from '../types.js';

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMResponse {
  content: string;
  usage: TokenUsage;
  model: string;
}

export async function callLLM(
  backbone: BackboneConfig,
  messages: LLMMessage[],
  options?: { temperature?: number; maxTokens?: number; signal?: AbortSignal }
): Promise<LLMResponse> {
  const temperature = options?.temperature ?? 0.7;
  const maxTokens = options?.maxTokens ?? 4096;
  const signal = options?.signal;

  if (backbone.provider === 'anthropic') {
    return callAnthropic(backbone, messages, temperature, maxTokens, signal);
  } else if (backbone.provider === 'openai') {
    return callOpenAI(backbone, messages, temperature, maxTokens, signal);
  } else {
    return callOpenRouter(backbone, messages, temperature, maxTokens, signal);
  }
}

async function callAnthropic(
  backbone: BackboneConfig,
  messages: LLMMessage[],
  temperature: number,
  maxTokens: number,
  signal?: AbortSignal
): Promise<LLMResponse> {
  const systemMsg = messages.find(m => m.role === 'system');
  const nonSystem = messages.filter(m => m.role !== 'system');

  const body: Record<string, unknown> = {
    model: backbone.model,
    max_tokens: maxTokens,
    temperature,
    messages: nonSystem.map(m => ({ role: m.role, content: m.content })),
  };
  if (systemMsg) {
    body.system = systemMsg.content;
  }

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': backbone.apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Anthropic API error ${res.status}: ${text}`);
  }

  const data = await res.json() as {
    content: Array<{ type: string; text: string }>;
    model: string;
    usage: { input_tokens: number; output_tokens: number };
  };

  return {
    content: data.content.map(c => c.text).join(''),
    model: data.model,
    usage: {
      prompt: data.usage.input_tokens,
      completion: data.usage.output_tokens,
      total: data.usage.input_tokens + data.usage.output_tokens,
    },
  };
}

async function callOpenAI(
  backbone: BackboneConfig,
  messages: LLMMessage[],
  temperature: number,
  maxTokens: number,
  signal?: AbortSignal
): Promise<LLMResponse> {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${backbone.apiKey}`,
    },
    body: JSON.stringify({
      model: backbone.model,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      temperature,
      max_tokens: maxTokens,
    }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`OpenAI API error ${res.status}: ${text}`);
  }

  const data = await res.json() as {
    choices: Array<{ message: { content: string } }>;
    model: string;
    usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  };

  return {
    content: data.choices[0].message.content,
    model: data.model,
    usage: {
      prompt: data.usage.prompt_tokens,
      completion: data.usage.completion_tokens,
      total: data.usage.total_tokens,
    },
  };
}

async function callOpenRouter(
  backbone: BackboneConfig,
  messages: LLMMessage[],
  temperature: number,
  maxTokens: number,
  signal?: AbortSignal
): Promise<LLMResponse> {
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${backbone.apiKey}`,
    },
    body: JSON.stringify({
      model: backbone.model,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      temperature,
      max_tokens: maxTokens,
    }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`OpenRouter API error ${res.status}: ${text}`);
  }

  const data = await res.json() as {
    choices: Array<{ message: { content: string } }>;
    model: string;
    usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  };

  return {
    content: data.choices[0].message.content,
    model: data.model,
    usage: {
      prompt: data.usage.prompt_tokens,
      completion: data.usage.completion_tokens,
      total: data.usage.total_tokens,
    },
  };
}
