const assert = require("node:assert/strict");
const { chromium } = require("playwright-core");

(async () => {
  const url = new URL(process.argv[2] || "http://127.0.0.1:8765/");
  assert.ok(["127.0.0.1", "localhost"].includes(url.hostname), "Use a local workbench");
  const browser = await chromium.launch({
    executablePath: process.env.INTERVIEW_CHROMIUM,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const logs = [];
    page.on("pageerror", (error) => logs.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") logs.push(message.text());
    });
    await page.goto(url.href);
    await page.waitForFunction(() => document.querySelector("#mode-pill").textContent === "Reference replay");
    const original = "How do I convert between tuples and lists?";
    const ask = async (question, calls) => {
      await page.locator("#question").fill(question);
      await page.locator("#ask-button").click();
      await page.waitForFunction((expected) => document.querySelector("#task-status").textContent === "Ready for your review" &&
        document.querySelector("#work-result").textContent.startsWith(expected + " scripted calls"), calls);
    };
    const approve = async () => {
      await page.locator("#approve").click();
      await page.waitForFunction(() => document.querySelector("#task-status").textContent === "Approved");
    };
    await ask(original, 3);
    await approve();
    await page.locator("#repeat").click();
    await page.waitForFunction(() => document.querySelector("#cache-result").textContent === "Exact answer reused" &&
      document.querySelector("#work-result").textContent.startsWith("0 scripted calls"));
    assert.match(await page.locator("#cache-receipt").innerText(), /source evidence matched/);
    await approve();
    await ask("How do I convert between tuples and lists without using a built-in constructor?", 3);
    await approve();
    await page.locator("#memory-button").click();
    await page.locator("#memory-dialog").waitFor({ state: "visible" });
    page.once("dialog", (dialog) => dialog.accept());
    await page.locator("#clear-memory").click();
    await page.waitForFunction(() => document.querySelector("#memory-message").textContent.includes("context cleared"));
    await page.keyboard.press("Escape");
    await ask(original, 3);
    await page.getByRole("link", { name: /Benchmark analysis/ }).click();
    await page.locator("#benchmark").waitFor({ state: "visible" });
    assert.equal(await page.locator("#exact-kpi").innerText(), "47.33%");
    assert.equal(await page.locator("#increment-kpi").innerText(), "21");
    await page.locator("#profile").selectOption("lower_repeat");
    assert.equal(await page.locator("#exact-kpi").innerText(), "14.75%");
    assert.equal(await page.locator("#increment-kpi").innerText(), "48");
    assert.equal(await page.locator("#quality-kpi").innerText(), "1 / 125");
    await page.locator("#path").selectOption("hit");
    assert.deepEqual(await page.locator("#latency tr").first().locator("td").allInnerTexts(), ["0", "—", "—"]);
    await page.locator("#coverage-arm").selectOption("exact_only");
    assert.deepEqual(await page.locator("#coverage tr").nth(1).locator("td").allInnerTexts(), ["59 / 60", "59 / 60"]);
    assert.match(await page.locator("#failure").innerText(), /cProfile/);
    assert.match(await page.locator("#paired-summary").innerText(), /16 of 400 requests, tied on 384, and used more on 0/);
    await page.locator("#dimension").selectOption("domain");
    await page.getByRole("button", { name: "asyncio coroutines", exact: true }).click();
    await page.locator("#outcome").selectOption("false_accept");
    assert.equal(await page.locator("#event-count").innerText(), "1–1 of 1 events");
    const falseAccept = page.getByRole("link", { name: "async_05:challenge_added_requirements" });
    assert.equal(await falseAccept.getAttribute("href"), "#case-study");
    assert.deepEqual(await page.locator("#paired-events tr td").allInnerTexts(), ["3", "3", "0", "3", "SEMANTIC · false accept"]);
    await page.locator("#event-search").fill("does-not-exist");
    assert.match(await page.locator("#event-count").innerText(), /No events/);
    for (const [label, filename] of [
      ["Download comparison", "comparison.csv"], ["Segment CSV", "segments.csv"], ["Event CSV", "paired-events.csv"],
    ]) {
      const download = page.waitForEvent("download");
      await page.getByRole("link", { name: new RegExp(label) }).click();
      assert.equal((await download).suggestedFilename(), filename);
    }
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    await page.getByRole("link", { name: /Workbench/ }).click();
    assert.equal(await page.getByRole("link", { name: /Benchmark analysis/ }).isVisible(), true);
    assert.deepEqual(logs, []);
    console.log("PASS: browser approval, 3→0 whole-answer repeat, changed requirement miss, context invalidation, request and domain drilldowns, CSV exports, mobile layout.");
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
