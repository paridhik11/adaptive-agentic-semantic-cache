"use strict";

const byId = (id) => document.getElementById(id);
const session = document.querySelector('meta[name="demo-session"]').content;
const turns = [];
const freshTimings = new Map();
const output = byId("conversation");
let previousQuestion = "";
let ready = false;
let busy = false;

function node(tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function time(ms) {
  return ms < 1000 ? `${ms.toFixed(1)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function appendMessage(title, text, type) {
  const article = node("article", "", type);
  article.append(node("h2", title), node("p", text, "message"));
  byId("start-hint").hidden = true;
  output.append(article);
  return article;
}

function buttons() {
  byId("send").disabled = !ready || busy;
  byId("repeat").disabled = !ready || busy || !previousQuestion;
  byId("follow-up").disabled = busy || turns.length === 0;
}

async function api(body) {
  const response = await fetch("/api/chat", {
    method: body ? "POST" : "GET",
    headers: { "X-Demo-Session": session, ...(body ? { "Content-Type": "application/json" } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  return [response.ok, await response.json()];
}

function stats(data) {
  byId("quota").textContent = `${data.used} / ${data.limit}`;
  byId("hits").textContent = data.hits;
  byId("model").textContent = `Live · ${data.model}`;
}

const bypassLabels = {
  cache_disabled_by_caller: "Exact reuse is disabled for this request.",
  context_requires_review: "The conversation was sent to the model for this follow-up.",
  personal_or_generative_request: "The policy sends personal or creative requests to the model.",
  not_rule_confirmed_static: "The policy sends requests without a static-topic match to the model.",
  freshness_requires_provider: "This request asks for current or personal information.",
  long_request: "The request exceeded the exact eligibility length.",
  cache_storage_unavailable: "Cached storage was unavailable. The model answered directly.",
};

async function send(question, includeHistory) {
  if (busy || !ready || !question.trim()) return;
  question = question.trim();
  busy = true;
  buttons();
  const history = includeHistory ? turns.slice(-6) : [];
  const error = byId("chat-error");
  error.hidden = true;
  appendMessage("You", question, "question");
  const pending = appendMessage("Trace", "Checking the cache…", "answer");
  try {
    const [ok, result] = await api({ question, history, cache: byId("cache-toggle").checked });
    if (!ok) {
      let detail = result.error || "Live request failed.";
      if (result.retry_at) detail += ` Retry after ${new Date(result.retry_at * 1000).toLocaleString()}.`;
      if (typeof result.provider_calls === "number") detail += ` Provider attempts for this request: ${result.provider_calls}.`;
      throw new Error(detail);
    }
    const names = { HIT: "Exact hit", MISS: "Fresh answer", BYPASS: "Fresh answer · cache bypassed" };
    pending.replaceChildren(node("h2", names[result.cache_status]), node("p", result.answer, "message"));
    const details = [
      `${result.provider_calls} provider call${result.provider_calls === 1 ? "" : "s"}`,
      `Response ${time(result.latency_ms)}`,
    ];
    if (result.provider_ms !== null) details.push(`Provider ${time(result.provider_ms)}`);
    if (result.generated_tokens !== null) details.push(`${result.generated_tokens} generated tokens`);
    if (!history.length && result.stored) {
      if (freshTimings.size >= 64) freshTimings.delete(freshTimings.keys().next().value);
      freshTimings.set(question, result.latency_ms);
    }
    if (result.cache_status === "HIT" && freshTimings.has(question)) {
      details.push(`Earlier fresh response ${time(freshTimings.get(question))}`);
    }
    const receipt = node("ul", "", "chat-receipt");
    receipt.append(...details.map((detail) => node("li", detail)));
    pending.append(receipt);
    if (result.cache_status === "BYPASS" && bypassLabels[result.cache_reason]) {
      pending.append(node("p", bypassLabels[result.cache_reason], "muted"));
    }
    stats(result);
    if (!includeHistory || result.answer.length > 8000) turns.length = 0;
    // Keep only complete usable turns. Oversize answers remain visible but are
    // excluded from the bounded follow-up payload.
    if (result.answer.length <= 8000) {
      turns.push({ role: "user", content: question }, { role: "assistant", content: result.answer });
      if (turns.length > 6) turns.splice(0, 2);
    }
    previousQuestion = question;
    byId("question").value = "";
    pending.scrollIntoView({ block: "nearest" });
  } catch (err) {
    pending.remove();
    error.textContent = err.message;
    error.hidden = false;
    api().then(([ok, result]) => { if (ok) stats(result); }).catch(() => {});
  } finally {
    busy = false;
    buttons();
  }
}

byId("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  send(byId("question").value, byId("follow-up").checked);
});
byId("repeat").addEventListener("click", () => {
  byId("follow-up").checked = false;
  send(previousQuestion, false);
});
api().then(([ok, result]) => {
  if (!ok) throw new Error(result.error);
  stats(result);
  ready = true;
  buttons();
}).catch((err) => {
  byId("model").textContent = "Disconnected";
  byId("chat-error").textContent = err.message;
  byId("chat-error").hidden = false;
});
