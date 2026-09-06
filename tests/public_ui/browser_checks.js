const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const BASE = "http://127.0.0.1:4173";
const IMAGE = path.join(ROOT, "static/images/example-landscape.jpg");
const FIXTURES = path.join(ROOT, "static/fixtures");
const EVIDENCE = path.join(ROOT, "test-results/public-ui");
fs.mkdirSync(EVIDENCE, { recursive: true });
const fixture = (name) => JSON.parse(fs.readFileSync(path.join(FIXTURES, name), "utf8"));
const results = [];
function check(name, condition) { if (!condition) throw new Error(`FAILED: ${name}`); results.push({ name, passed: true }); }

async function mockAnalysis(page, response, status = 200, delay = 0, counter = null) {
  await page.route("**/api/v1/analyses", async (route) => {
    if (counter) counter.count += 1;
    if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
    await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(response) });
  });
  await page.route("**/api/v1/feedback", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(fixture("feedback-success.json")) }));
}

async function chooseAndSubmit(page) {
  await page.locator("#image-input").setInputFiles(IMAGE);
  await page.locator("[data-submit]").click();
}

async function errorJourney(browser, status, fixtureName, expected) {
  const page = await browser.newPage();
  await mockAnalysis(page, fixture(fixtureName), status);
  await page.goto(BASE);
  await chooseAndSubmit(page);
  await page.locator("[data-error-state]").waitFor({ state: "visible" });
  check(`HTTP ${status} error`, (await page.locator("[data-error-title]").textContent()).includes(expected));
  check(`HTTP ${status} never says ready`, !(await page.locator("body").textContent()).includes("critique ready"));
  await page.close();
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await mockAnalysis(page, fixture("analysis-success.json"));
    await page.goto(BASE);
    check("landing page", await page.locator("h1").count() === 1 && (await page.locator("h1").textContent()).includes("Look closer"));
    check("semantic landmarks", await page.locator(".site-header").count() === 1 && await page.locator("main").count() === 1 && await page.locator(".site-footer").count() === 1);
    await page.keyboard.press("Tab"); check("skip link keyboard", await page.locator(".skip-link").evaluate((el) => el === document.activeElement));
    await page.locator("#critique").scrollIntoViewIfNeeded();
    await page.locator("[data-drop-zone]").focus(); check("keyboard reaches upload", await page.locator("[data-drop-zone]").evaluate((el) => el === document.activeElement));
    await page.locator("#image-input").setInputFiles(IMAGE);
    check("preview renders", await page.locator("[data-preview]").getAttribute("src").then(Boolean));
    check("selected image enables submit", await page.locator("[data-submit]").isEnabled());
    await page.locator("[data-submit]").click();
    await page.locator("[data-result]").waitFor({ state: "visible" });
    check("successful critique", (await page.locator("[data-critique]").textContent()).includes("road gives the eye"));
    check("evidence rendering", (await page.locator("[data-recognition]").textContent()).includes("mountain valley"));
    check("measured signals visible", await page.locator("[data-measured-signals]").isVisible());
    check("measured signal count", await page.locator("[data-signals-list] > div").count() === 4);
    check("limitations rendering", await page.locator("[data-limitations] li").count() === 2);
    check("terminal focus moved", await page.locator("[data-result]").evaluate((el) => el === document.activeElement));
    await page.locator('[data-feedback-value="useful"]').click();
    await page.waitForFunction(() => document.querySelector("[data-feedback-status]")?.textContent.includes("attached"));
    check("feedback controls", (await page.locator("[data-feedback-status]").textContent()).includes("attached"));
    await page.locator("[data-evidence-toggle]").click();
    check("evidence disclosure", await page.locator("[data-evidence-toggle]").getAttribute("aria-expanded") === "false");
    await page.locator("[data-evidence-toggle]").click();
    await page.locator(".skip-link").evaluate((element) => { element.blur(); element.style.display = "none"; });
    await page.screenshot({ path: path.join(EVIDENCE, "result-1280.png"), fullPage: true });
    await page.close();

    const invalid = await browser.newPage(); await invalid.goto(BASE); await invalid.locator("#image-input").setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("not an image") });
    check("invalid file", (await invalid.locator("[data-file-error]").textContent()).includes("JPEG")); await invalid.close();
    await errorJourney(browser, 400, "error-400.json", "not a usable");
    await errorJourney(browser, 413, "error-413.json", "too large");
    await errorJourney(browser, 429, "error-429.json", "capacity");
    await errorJourney(browser, 504, "error-timeout.json", "took too long");
    await errorJourney(browser, 503, "error-503.json", "unavailable");
    await errorJourney(browser, 500, "error-500.json", "could not finish");

    const network = await browser.newPage(); await network.route("**/api/v1/analyses", (route) => route.abort("failed")); await network.goto(BASE); await chooseAndSubmit(network); await network.locator("[data-error-state]").waitFor({ state: "visible" });
    check("network error", (await network.locator("[data-error-title]").textContent()).includes("could not be reached")); await network.close();

    const retry = await browser.newPage(); let attempts = 0; await retry.route("**/api/v1/analyses", async (route) => { attempts++; await route.fulfill({ status: attempts === 1 ? 500 : 200, contentType: "application/json", body: JSON.stringify(attempts === 1 ? fixture("error-500.json") : fixture("analysis-success.json")) }); }); await retry.goto(BASE); await chooseAndSubmit(retry); await retry.locator("[data-error-state]").waitFor({ state: "visible" }); await retry.locator("[data-retry]").click(); await retry.locator("[data-result]").waitFor({ state: "visible" }); check("retry", attempts === 2); await retry.close();

    const duplicate = await browser.newPage(); const counter = { count: 0 }; await mockAnalysis(duplicate, fixture("analysis-success.json"), 200, 700, counter); await duplicate.goto(BASE); await duplicate.locator("#image-input").setInputFiles(IMAGE); await duplicate.locator("#analysis-form").evaluate((form) => { form.requestSubmit(); form.requestSubmit(); }); await duplicate.locator("[data-result]").waitFor({ state: "visible" }); check("duplicate submit prevention", counter.count === 1); await duplicate.close();

    const cancelled = await browser.newPage(); await cancelled.route("**/api/v1/analyses", async (route) => { await new Promise((resolve) => setTimeout(resolve, 2000)); await route.abort(); }); await cancelled.goto(BASE); await chooseAndSubmit(cancelled); await cancelled.locator("[data-cancel]").click(); await cancelled.locator("[data-error-state]").waitFor({ state: "visible" }); check("cancel", (await cancelled.locator("[data-error-title]").textContent()).includes("cancelled")); await cancelled.close();

    const malformed = await browser.newPage(); await mockAnalysis(malformed, fixture("malformed-response.json")); await malformed.goto(BASE); await chooseAndSubmit(malformed); await malformed.locator("[data-error-state]").waitFor({ state: "visible" }); check("malformed response", (await malformed.locator("[data-error-title]").textContent()).includes("incomplete")); await malformed.close();

    const empty = await browser.newPage(); await mockAnalysis(empty, fixture("analysis-empty-evidence.json")); await empty.goto(BASE); await chooseAndSubmit(empty); await empty.locator("[data-result]").waitFor({ state: "visible" }); check("empty evidence", await empty.locator("[data-evidence-empty]").isVisible()); await empty.close();

    const disabled = await browser.newPage(); await mockAnalysis(disabled, fixture("analysis-grounding-disabled.json")); await disabled.goto(BASE); await chooseAndSubmit(disabled); await disabled.locator("[data-result]").waitFor({ state: "visible" }); check("grounding disabled", (await disabled.locator("[data-grounding]").textContent()).includes("disabled")); await disabled.close();

    const xss = await browser.newPage(); await mockAnalysis(xss, fixture("analysis-xss.json")); await xss.goto(BASE); await chooseAndSubmit(xss); await xss.locator("[data-result]").waitFor({ state: "visible" }); check("XSS remains inert", await xss.evaluate(() => window.__xss === undefined)); check("XSS literal visible", (await xss.locator("[data-critique]").textContent()).includes("<img src=x")); await xss.close();

    const reduced = await browser.newPage({ reducedMotion: "reduce" }); await reduced.goto(BASE); check("reduced motion", await reduced.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches)); await reduced.close();

    const privacy = await browser.newPage(); await privacy.goto(`${BASE}/privacy`); check("privacy page", await privacy.locator("h1").count() === 1 && (await privacy.locator("main").textContent()).includes("No public continuity")); await privacy.close();

    for (const width of [320, 375, 768, 1280]) {
      const mobile = await browser.newPage({ viewport: { width, height: 900 } });
      await mockAnalysis(mobile, fixture("analysis-success.json"), 200, 1500);
      await mobile.goto(BASE);
      check(`no overflow ${width}`, await mobile.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth));
      await mobile.screenshot({ path: path.join(EVIDENCE, `landing-${width}.png`), fullPage: true });
      await chooseAndSubmit(mobile);
      await mobile.locator("[data-progress]").waitFor({ state: "visible" });
      await mobile.screenshot({ path: path.join(EVIDENCE, `loading-${width}.png`) });
      await mobile.locator("[data-result]").waitFor({ state: "visible" });
      check(`result signals ${width}`, await mobile.locator("[data-measured-signals]").isVisible());
      check(`result no overflow ${width}`, await mobile.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth));
      await mobile.locator("[data-result]").screenshot({ path: path.join(EVIDENCE, `critique-${width}.png`) });
      await mobile.locator('[data-feedback-value="useful"]').click();
      await mobile.waitForFunction(() => document.querySelector("[data-feedback-status]")?.textContent.includes("attached"));
      check(`feedback ${width}`, true);
      await mobile.close();
      for (const status of [503, 429]) {
        const failure = await browser.newPage({ viewport: { width, height: 900 } });
        await mockAnalysis(failure, fixture(`error-${status}.json`), status);
        await failure.goto(BASE);
        await chooseAndSubmit(failure);
        await failure.locator("[data-error-state]").waitFor({ state: "visible" });
        check(`failure ${status} ${width}`, !(await failure.locator("[data-result]").isVisible()));
        await failure.locator("[data-error-state]").screenshot({ path: path.join(EVIDENCE, `error-${status}-${width}.png`) });
        await failure.close();
      }
    }
    fs.writeFileSync(path.join(EVIDENCE, "browser-checks.json"), JSON.stringify({ passed: results.length, checks: results }, null, 2));
    console.log(JSON.stringify({ passed: results.length, evidence: EVIDENCE }));
  } finally { await browser.close(); }
})().catch((error) => { console.error(error.stack || error); process.exit(1); });
