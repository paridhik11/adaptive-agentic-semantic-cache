"use strict";

const byId = (id) => document.getElementById(id);
const session = document.querySelector('meta[name="demo-session"]').content;
const turns = [];
const freshTimings = new Map();
const output = byId("conversation");
let previousQuestion = "";
let ready = false;
let busy = false;

const simpleFollowUp = /^(?:why|how so|what do you mean|(?:please )?(?:explain|expand on|summarize|clarify) (?:that|this|it)|(?:please )?(?:give|show)(?: me)? (?:an? (?:worked |small |simple )?)?example)[?.!]*$/i;

function node(tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function time(ms) {
  return ms < 1000 ? `${ms.toFixed(1)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function inlineText(element, text) {
  let offset = 0;
  for (const match of text.matchAll(/`([^`\n]+)`|\*\*([^*\n]+)\*\*/g)) {
    element.append(document.createTextNode(text.slice(offset, match.index)));
    element.append(match[1] ? node("code", match[1]) : node("strong", match[2]));
    offset = match.index + match[0].length;
  }
  element.append(document.createTextNode(text.slice(offset)));
  return element;
}

function answerContent(text) {
  const content = node("div", "", "answer-content");
  let list = null;
  let paragraph = [];
  const flush = () => {
    if (paragraph.length) content.append(inlineText(node("p", "", "message"), paragraph.join("\n")));
    paragraph = [];
  };
  const lines = text.split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    const item = line.match(/^\s*(\d+\.|[-*+])\s+(.+)$/);
    if (item) {
      flush();
      const tag = /\d/.test(item[1][0]) ? "ol" : "ul";
      if (!list || list.localName !== tag) {
        list = node(tag, "");
        if (tag === "ol") list.start = Number.parseInt(item[1], 10);
        content.append(list);
      }
      list.append(inlineText(node("li", ""), item[2]));
      continue;
    }
    list = null;
    if (/^\s*```/.test(line)) {
      flush();
      const code = [];
      while (++index < lines.length && !/^\s*```\s*$/.test(lines[index])) code.push(lines[index]);
      const block = node("pre", "");
      block.append(node("code", code.join("\n")));
      content.append(block);
    } else if (/^#{1,6} /.test(line)) {
      flush();
      content.append(inlineText(node("h3", ""), line.replace(/^#+ /, "")));
    } else if (!line.trim()) {
      flush();
    } else {
      paragraph.push(line);
    }
  }
  flush();
  return content;
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
  byId("question").readOnly = busy;
  byId("cache-toggle").disabled = busy;
}

async function api(body) {
  const response = await fetch("/api/chat", {
    method: body ? "POST" : "GET",
    headers: { "X-Demo-Session": session, ...(body ? { "Content-Type": "application/json" } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  if (response.status === 401) {
    ready = false;
    byId("model").textContent = "Refresh to reconnect";
    throw new Error("The server session changed. Refresh this page to reconnect.");
  }
  return [response.ok, await response.json().catch(() => ({error: "Trace could not read the server response. Check the server terminal."}))];
}

function stats(data) {
  byId("quota").textContent = `${data.used ?? "—"} / ${data.limit}`;
  byId("hits").textContent = data.hits;
  byId("hit-label").textContent = `${data.exact_hits} exact · ${data.semantic_hits} semantic`;
  byId("model").textContent = `Live · ${data.model} · ${data.semantic_enabled ? "local models on" : "local reuse on"}`;
}

const bypassLabels = {
  cache_disabled_by_caller: "You turned off cache reuse for this request.",
  context_requires_review: "The conversation was sent to the model for this follow-up.",
  personal_or_generative_request: "The policy sends personal or creative requests to the model.",
  not_rule_confirmed_static: "The policy sends requests without a static-topic match to the model.",
  freshness_requires_provider: "This request asks for current or personal information.",
  long_request: "The request exceeded the exact eligibility length.",
  cache_storage_unavailable: "Cached storage was unavailable. The model answered directly.",
};

async function send(question, includeHistory, {clearDraft = true, autoContext = false} = {}) {
  if (busy || !ready || !question.trim()) return;
  question = question.trim();
  busy = true;
  buttons();
  const history = includeHistory ? turns.slice(-6) : [];
  const error = byId("chat-error");
  error.hidden = true;
  appendMessage("You", question, "question");
  const pending = appendMessage("Trace", "Checking the cache or waiting for the model…", "answer");
  try {
    const body = {question, history, cache: byId("cache-toggle").checked};
    if (autoContext && !includeHistory && turns.length) {
      body.previous_question = turns[turns.length - 2].content;
    }
    const [ok, result] = await api(body);
    if (!ok) {
      let detail = result.error || "Live request failed.";
      if (result.retry_at) detail += ` Retry after ${new Date(result.retry_at * 1000).toLocaleString()}.`;
      if (typeof result.provider_calls === "number") detail += ` Provider attempts for this request: ${result.provider_calls}.`;
      throw new Error(detail);
    }
    const names = { CLARIFY: "Clarification needed", HIT: "Exact hit", MISS: "Fresh answer", BYPASS: "Fresh answer · cache bypassed" };
    const label = result.cache_tier === "SEMANTIC" ? "Semantic hit" : names[result.cache_status];
    pending.replaceChildren(node("h2", label), answerContent(result.answer));
    const details = [
      `${result.provider_calls} provider call${result.provider_calls === 1 ? "" : "s"}`,
      `Response ${time(result.latency_ms)}`,
    ];
    if (result.provider_ms !== null) details.push(`Provider ${time(result.provider_ms)}`);
    if (result.generated_tokens !== null) details.push(`${result.generated_tokens} generated tokens`);
    const effectiveQuestion = result.resolved_question || question;
    if (result.resolved_question) {
      pending.append(node("p", `Used the preceding question to resolve “${result.resolved_question}”`, "muted"));
    }
    if (!history.length && result.stored) {
      if (freshTimings.size >= 64) freshTimings.delete(freshTimings.keys().next().value);
      freshTimings.set(effectiveQuestion, result.latency_ms);
    }
    const source = result.semantic_match?.matched_question ?? effectiveQuestion;
    if (result.cache_status === "HIT" && freshTimings.has(source)) {
      details.push(`Earlier fresh response ${time(freshTimings.get(source))}`);
    }
    if (result.semantic_match) {
      const { similarity, lexical_similarity } = result.semantic_match;
      const detail = result.semantic_decision === "intent_verified" ? "same subject and requested fact or calculation" :
        result.semantic_decision === "keyword_verified" ? "same ordered keywords and request type" :
        result.semantic_decision === "wording_verified" ? "same request with different framing" :
        result.semantic_decision === "comparison_verified" ? "same comparison" :
        Number.isFinite(lexical_similarity) ? `same defined topic · text similarity ${lexical_similarity.toFixed(2)}` :
        similarity === null ? "same defined topic" : `cosine similarity ${similarity.toFixed(2)}`;
      pending.append(node("p", `Matched saved question “${source}” · ${detail}`, "muted"));
    }
    const receipt = node("ul", "", "chat-receipt");
    receipt.append(...details.map((detail) => node("li", detail)));
    pending.append(receipt);
    if (result.cache_status === "BYPASS" && bypassLabels[result.cache_reason]) {
      pending.append(node("p", bypassLabels[result.cache_reason], "muted"));
    }
    const semanticReasons = {
      no_close_candidate: "No eligible saved question was similar enough.",
      ambiguous_candidates: "Several saved questions matched closely. Trace asked the model.",
      meaning_not_verified: "Local checks could not verify that the questions meant the same thing.",
      answer_not_verified: "Local checks could not verify that the saved answer fits this question.",
      local_check_unavailable: "Local semantic verification was unavailable. Trace asked the model.",
    };
    if (semanticReasons[result.semantic_decision]) {
      pending.append(node("p", semanticReasons[result.semantic_decision], "muted"));
    }
    stats(result);
    byId("last-time").textContent = time(result.latency_ms);
    if (!includeHistory || result.answer.length > 8000) turns.length = 0;
    // Keep only complete usable turns. Oversize answers remain visible but are
    // excluded from the bounded follow-up payload.
    if (result.answer.length <= 8000) {
      turns.push({ role: "user", content: effectiveQuestion }, { role: "assistant", content: result.answer });
      if (turns.length > 6) turns.splice(0, 2);
    }
    previousQuestion = effectiveQuestion;
    if (clearDraft) byId("question").value = "";
    pending.scrollIntoView({ block: "nearest" });
  } catch (err) {
    pending.replaceChildren(node("h2", "Request stopped"), node("p", err.message, "message"));
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
  const question = byId("question").value.trim();
  const context = byId("follow-up").value;
  send(question, turns.length > 0 && (context === "follow-up" ||
    (context === "auto" && simpleFollowUp.test(question))), {autoContext: context === "auto"});
});
byId("repeat").addEventListener("click", () => {
  send(previousQuestion, false, {clearDraft: false});
});
byId("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    byId("chat-form").requestSubmit();
  }
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
