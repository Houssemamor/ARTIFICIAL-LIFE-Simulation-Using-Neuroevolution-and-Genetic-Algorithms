const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];

let drawCalls = 0;
const ctxStub = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (...a) => { drawCalls++; }),
  set: (t, k, v) => { t[k] = v; return true; },
});
class El {
  constructor(id) {
    this.id = id; this.innerHTML = ""; this.children = []; this.style = {};
    this.dataset = {}; this.attrs = {}; this.width = 640; this.height = 460;
    this.clientWidth = 800; this.value = "0"; this.max = "1"; this.open = false;
    this.className = ""; this.textContent = "";
    this.classList = { toggle() {}, remove() {}, add() {} };
  }
  addEventListener() {} setAttribute(k, v) { this.attrs[k] = v; }
  appendChild(c) { this.children.push(c); return c; }
  getContext() { return (this._ctx ??= ctxStub()); }
  querySelector() { return (this._canvas ??= new El("canvas")); }
}
const els = {};
const byId = id => (els[id] ??= new El(id));
let rafCallback = null;
const context = {
  console, Math, JSON, Map, Set, Promise, Number, Object, Array, String, Infinity,
  setTimeout,
  window: { matchMedia: () => ({ matches: false }), devicePixelRatio: 1, addEventListener() {} },
  document: {
    hidden: false, getElementById: byId,
    createElement: () => new El("n"),
    querySelectorAll: () => [], querySelector: () => new El("q"), addEventListener() {},
  },
  requestAnimationFrame: cb => { rafCallback = cb; },
  fetch: () => { throw new Error("fetch must not be used in static mode"); },
};
context.globalThis = context;
vm.createContext(context);

(async () => {
  vm.runInContext(js, context);
  await new Promise(r => setTimeout(r, 100));
  const run = code => vm.runInContext(code, context);

  const scenarios = run("state.recording.scenarios.length");
  let frameDraws = 0;
  for (let s = 0; s < scenarios; s++) {
    for (const brain of ["untrained", "evolved"]) {
      run(`state.scenario = ${s}; state.brain = '${brain}'; refreshShowcase();`);
      const frames = run("state.run.frames.length");
      // Every frame and several sub-step positions, including the last frame.
      for (let f = 0; f < frames; f++) {
        for (const frac of [0, 0.35, 0.9]) {
          run(`state.pos = ${Math.min(f + frac, frames - 1)}; render();`);
          frameDraws++;
        }
      }
      // Exercise the playback loop and looping.
      run("state.playing = true; state.visible = true;");
      for (let k = 1; k <= 50; k++) rafCallback(k * 400);
      const panels = run("state.panels.length");
      console.log(`scenario ${s} (${run("currentScenario().id")}) ${brain}: frames=${frames} panels=${panels}`);
    }
  }
  const findings = byId("findings-grid").innerHTML;
  console.log("renders:", frameDraws, "draw calls:", drawCalls);
  console.log("findings cards:", (findings.match(/<article/g) || []).length,
    "| badges:", [...findings.matchAll(/v-(\w+)">/g)].map(m => m[1]).join(","));
  console.log("charts svg:", (findings.match(/<svg/g) || []).length, "strip:", findings.includes('class="strip"'));
  console.log("note:", byId("run-note").innerHTML.slice(0, 220));
})().catch(e => { console.error("FAIL", e); process.exit(1); });
