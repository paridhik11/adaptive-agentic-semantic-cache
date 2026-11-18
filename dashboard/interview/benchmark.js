"use strict";

const byId = (id) => document.getElementById(id);
const put = (id, value) => { byId(id).textContent = value; };
const number = new Intl.NumberFormat("en-US");
const decimal = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmt = (value) => value === null ? "—" : decimal.format(value);
const pct = (value) => value === null ? "—" : fmt(value) + "%";
const arms = {
  no_cache: "No cache",
  exact_only: "Exact only",
  verified_semantic: "Guarded semantic",
  shadow_semantic: "Shadow semantic",
};
const categories = {
  seed: "Seed", exact_repeat: "Exact repeat", natural_paraphrase: "Paraphrase", adversarial_challenge: "Challenge",
};

function row(label, columns) {
  const item = document.createElement("tr");
  const name = document.createElement("th");
  name.scope = "row";
  name.textContent = label;
  item.append(name);
  for (const value of columns) {
    const cell = document.createElement("td");
    cell.textContent = value;
    item.append(cell);
  }
  return item;
}

function challengeText(challenge) {
  const [low, high] = challenge.exact_95_ci_pct;
  return `${pct(challenge.pct)} · 95% CI ${pct(low)}–${pct(high)}`;
}

function showComparison(profile, isRepeatHeavy) {
  const baseline = profile.arms.no_cache;
  const exact = profile.arms.exact_only;
  const guarded = profile.arms.verified_semantic;
  const increment = profile.semantic_increment;
  const challenge = guarded.adversarial_false_accepts;
  put("profile-scope", `${profile.events_per_arm} events per configuration · ${profile.source_groups} source groups · ${profile.intents} base intents`);
  put("exact-kpi", pct(exact.calls_saved_pct));
  put("exact-detail", `${number.format(exact.calls_saved)} of ${number.format(baseline.calls)} baseline calls`);
  put("increment-kpi", number.format(increment.calls_saved));
  put("increment-detail", `${pct(increment.pct_of_baseline_calls)} of baseline calls · ${increment.calls_saved_on_eligible_hits} on workload-eligible hits`);
  put("quality-kpi", `${challenge.hits} / ${challenge.n}`);
  put("quality-detail", challengeText(challenge));
  const latencyDelta = exact.latency_ms.all.p50 - baseline.latency_ms.all.p50;
  const deltaText = latencyDelta >= 0 ? "higher" : "lower";
  put("takeaway", `Exact reuse captured ${pct(100 * exact.calls_saved / guarded.calls_saved)} of the guarded configuration’s gross call savings. Exact-only median latency was ${fmt(Math.abs(latencyDelta))} ms ${deltaText} than no cache in this replay.`);
  put("quality-note", isRepeatHeavy
    ? "Adversarial results on this workload are development regression evidence."
    : "One guarded hit saved three calls on an ineligible challenge. Work avoided on that event was an incorrect substitution.");
  put("source", profile.source);
  put("sha", profile.sha256);

  const chart = [];
  const comparison = [];
  for (const [key, name] of Object.entries(arms)) {
    const data = profile.arms[key];
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = name;
    const meter = document.createElement("meter");
    meter.max = baseline.calls;
    meter.value = data.calls;
    meter.className = key;
    meter.setAttribute("aria-label", `${name} scripted calls`);
    const total = document.createElement("strong");
    total.textContent = number.format(data.calls);
    item.append(label, meter, total);
    chart.push(item);
    const adversarial = data.adversarial_false_accepts;
    const review = data.answer_review;
    comparison.push(row(name, [
      number.format(data.calls), `${number.format(data.calls_saved)} · ${pct(data.calls_saved_pct)}`,
      number.format(data.synthetic_tokens), pct(data.synthetic_tokens_saved_pct),
      `${adversarial.hits} / ${adversarial.n}`, `${review.reviewed_accepts} / ${review.total_accepts}`,
    ]));
  }
  byId("chart").replaceChildren(...chart);
  byId("comparison").replaceChildren(...comparison);
}

function showLatency(profile) {
  const path = byId("path").value;
  const rows = Object.entries(arms).map(([key, label]) => {
    const values = profile.arms[key].latency_ms[path];
    return row(label, [values.n, fmt(values.p50), fmt(values.p95)]);
  });
  byId("latency").replaceChildren(...rows);
}

function showCoverage(profile) {
  const coverage = profile.arms[byId("coverage-arm").value].coverage;
  const rows = Object.entries(categories).map(([key, label]) => {
    const values = coverage[key];
    return row(label, [`${values.hits} / ${values.requests}`, `${values.eligible_hits} / ${values.eligible}`]);
  });
  byId("coverage").replaceChildren(...rows);
}

function showFailure(failure) {
  put("event-id", failure.event_id);
  const fields = {
    Request: failure.question,
    "Retrieved prompt": failure.matched_question,
    "Returned answer": failure.actual_response_answer,
    "Missed requirement": "The answer leaves out benchmarking with cProfile.",
    Decision: `Guarded semantic accepted at tier ${failure.tier} (${failure.provider_calls} scripted calls). Exact only fell back (${failure.exact_only_calls} calls).`,
  };
  for (const [label, value] of Object.entries(fields)) {
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = value;
    byId("failure").append(term, detail);
  }
}

function segmentName(segment) {
  return categories[segment] || segment.replaceAll("_", " ");
}

function showSegments(profile, selection, selectSegment) {
  const increment = profile.semantic_increment;
  put("paired-summary", `Guarded used fewer calls on ${increment.events_with_fewer_calls} of ${profile.events_per_arm} requests, tied on ${increment.events_with_same_calls}, and used more on ${increment.events_with_more_calls}.`);
  const groups = profile.segments.filter((item) => item.dimension === byId("dimension").value);
  byId("segment-rows").replaceChildren(...groups.map((item) => {
    const tr = row(segmentName(item.segment), [
      item.requests, item.exact_calls_avoided, item.guarded_increment,
      item.challenges ? `${item.guarded_false_accepts} / ${item.challenges}` : "—",
    ]);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "segment-button";
    button.textContent = segmentName(item.segment);
    button.setAttribute("aria-pressed", String(item.segment === selection));
    button.addEventListener("click", () => {
      for (const other of byId("segment-rows").querySelectorAll("button")) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      selectSegment(item.segment);
    });
    tr.firstChild.replaceChildren(button);
    return tr;
  }));
}

function eventInspector(events, selection, page) {
  const key = byId("dimension").value;
  const outcome = byId("outcome").value;
  const search = byId("event-search").value.trim().toLowerCase();
  const rows = events.filter((event) => event[key] === selection && event.event_id.toLowerCase().includes(search))
    .filter((event) => ({
      all: true, fewer: event.guarded_increment > 0, same: event.guarded_increment === 0,
      more: event.guarded_increment < 0, false_accept: event.guarded_false_accept === 1,
    })[outcome]);
  const pageSize = 10;
  const start = page * pageSize;
  put("event-context", `Showing ${segmentName(selection)}. Each row pairs the same request across all four configurations.`);
  put("event-count", rows.length ? `${start + 1}–${Math.min(start + pageSize, rows.length)} of ${rows.length} events` : "No events match these filters.");
  byId("prev-events").disabled = page === 0;
  byId("next-events").disabled = start + pageSize >= rows.length;
  byId("paired-events").replaceChildren(...rows.slice(start, start + pageSize).map((event) => {
    const entry = row(event.event_id, [event.no_cache_calls, event.exact_calls, event.guarded_calls,
      event.shadow_calls, event.guarded_false_accept ? `${event.guarded_tier} · false accept` : event.guarded_tier]);
    if (event.guarded_false_accept) {
      entry.lastChild.classList.add("flagged");
      const link = document.createElement("a");
      link.href = "#case-study";
      link.textContent = event.event_id;
      entry.firstChild.replaceChildren(link);
    }
    return entry;
  }));
}

async function start() {
  const responses = await Promise.all([fetch("report.json"), fetch("paired-events.json")]);
  if (responses.some((response) => !response.ok)) throw new Error("Report could not be loaded");
  const [report, paired] = await Promise.all(responses.map((response) => response.json()));
  const events = Object.fromEntries(Object.entries(paired).map(([workload, tables]) => [
    workload, tables.rows.map((values) => Object.fromEntries(tables.columns.map((col, index) => [col, values[index]]))),
  ]));
  put("scope", `${number.format(report.executions)} offline executions · ${number.format(report.events_replayed)} matched events · 4 configurations`);
  const profile = () => report.profiles[byId("profile").value];
  let selection;
  let page = 0;
  const showEvents = () => eventInspector(events[byId("profile").value], selection, page);
  const resetPage = () => {
    page = 0;
    showEvents();
  };
  const selectSegment = (segment) => {
    selection = segment;
    byId("outcome").value = "all";
    byId("event-search").value = "";
    resetPage();
  };
  const refreshSegments = () => {
    selectSegment(profile().segments.find((item) => item.dimension === byId("dimension").value).segment);
    showSegments(profile(), selection, selectSegment);
  };
  const refresh = () => {
    showComparison(profile(), byId("profile").value === "repeat_rich");
    showLatency(profile());
    showCoverage(profile());
    refreshSegments();
  };
  byId("dimension").addEventListener("change", refreshSegments);
  byId("outcome").addEventListener("change", resetPage);
  byId("event-search").addEventListener("input", resetPage);
  byId("prev-events").addEventListener("click", () => { page -= 1; showEvents(); });
  byId("next-events").addEventListener("click", () => { page += 1; showEvents(); });
  byId("profile").addEventListener("change", refresh);
  byId("path").addEventListener("change", () => showLatency(profile()));
  byId("coverage-arm").addEventListener("change", () => showCoverage(profile()));
  showFailure(report.profiles.lower_repeat.failures[0]);
  refresh();
  byId("benchmark").hidden = false;
}

start().catch(() => {
  put("scope", "Offline campaign evidence");
  put("error", "Could not load the benchmark data. Open this page from the local workbench and regenerate the report if the data is missing.");
  byId("error").hidden = false;
});
