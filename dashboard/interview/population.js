"use strict";

const byId = (id) => document.getElementById(id);
const amount = (value) => value.toFixed(1);
const points = (value) => `${value.toFixed(2)} pp`;
const policyLabels = {
  exact: "Exact only",
  guarded: "Current guarded",
  reporting: "Reporting check",
  context: "Trusted context",
};

function row(label, values) {
  const element = document.createElement("tr");
  const heading = document.createElement("th");
  heading.scope = "row";
  heading.textContent = label;
  element.append(heading);
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    element.append(cell);
  }
  return element;
}

async function start() {
  const response = await fetch("workload-population.json");
  if (!response.ok) throw new Error("Could not read population data");
  const { totals, gap, scenarios } = await response.json();
  byId("savings-gap").textContent = points(totals.repeat_heavy_pct - totals.broader_pct);
  byId("composition-gap").textContent = points(totals.mix_pp);
  byId("rate-gap").textContent = points(totals.within_segment_pp);
  byId("gap-by-type").replaceChildren(...gap.map((segment) => row(
    segment.request_type.replaceAll("_", " "),
    [segment.repeat_heavy_n, segment.broader_n, points(segment.mix_gap_pp), points(segment.within_segment_gap_pp)],
  )));

  const repeatControl = byId("repeat-share");
  const contextControl = byId("trusted-share");
  function updateProjection() {
    const scenario = `repeat_${repeatControl.value}_context_${contextControl.value}`;
    const policies = Object.fromEntries(
      scenarios.filter((result) => result.scenario === scenario).map((result) => [result.policy, result]),
    );
    byId("projected-outcomes").replaceChildren(...Object.entries(policyLabels).map(([policy, label]) => {
      const result = policies[policy];
      return row(label, [
        amount(result.suitable_avoided_per_1000),
        amount(result.unsuitable_per_1000),
        amount(result.fallback_per_1000),
      ]);
    }));
    byId("projection-summary").textContent =
      `At this mix, trusted context projects ${amount(policies.context.suitable_avoided_per_1000)} suitable calls avoided and ${amount(policies.context.fallback_per_1000)} fallbacks per 1,000 requests.`;
  }
  repeatControl.addEventListener("change", updateProjection);
  contextControl.addEventListener("change", updateProjection);
  updateProjection();
  byId("population-status").hidden = true;
  byId("population").hidden = false;
}

start().catch(() => {
  byId("population-status").textContent = "Population data could not be loaded. Regenerate it with the report command below.";
});
