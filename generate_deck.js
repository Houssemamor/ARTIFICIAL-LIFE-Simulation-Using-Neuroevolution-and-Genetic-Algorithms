// Slide deck for the neuroevolution project (Phase 9).
// Run: node generate_deck.js
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5

// Palette: deep-space night world + evolutionary accent
const INK = "0E1420";      // dominant dark
const PANEL = "182238";    // card background
const TEXT = "EAF0FF";    // primary text
const MUTED = "8FA3C8";   // secondary text
const CYAN = "35C4D8";    // accent 1 (evolution / learning)
const AMBER = "F2A33C";    // accent 2 (predation)
const GREEN = "3FD18B";    // positive results
const RED = "E4576B";      // caveats
const FONT_H = "Cambria";
const FONT_B = "Calibri";

function darkSlide() {
  const slide = pres.addSlide();
  slide.background = { color: INK };
  return slide;
}
function title(slide, text, sub) {
  slide.addText(text, { x: 0.6, y: 0.5, w: 12.1, h: 1.0, fontFace: FONT_H,
    fontSize: 38, bold: true, color: TEXT, margin: 0 });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.55, w: 12.1, h: 0.6,
    fontFace: FONT_B, fontSize: 17, color: CYAN, italic: true, margin: 0 });
}
function bullets(slide, items, opts = {}) {
  slide.addText(items.map((t, i) => ({
    text: t.text, options: {
      bullet: t.sub ? { code: "2013", indent: 14 } : { code: "25AA", indent: 16 },
      indentLevel: t.sub ? 1 : 0, breakLine: true,
      color: t.color || (t.sub ? MUTED : TEXT),
      fontSize: t.size || (t.sub ? 15 : 17),
      paraSpaceAfter: t.sub ? 4 : 10, bold: !!t.bold,
    } })), { x: 0.7, y: opts.y || 2.3, w: opts.w || 7.6, h: opts.h || 4.4,
      fontFace: FONT_B, valign: "top", margin: 0 });
}
function stat(slide, x, y, number, label, color) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w: 3.7, h: 1.7,
    fill: { color: PANEL }, line: { color: "24304A", width: 1 },
    rectRadius: 0.08, shadow: { type: "outer", color: "000000", blur: 6,
      offset: 2, angle: 90, opacity: 0.4 } });
  slide.addText(number, { x: x + 0.2, y: y + 0.12, w: 3.3, h: 0.9,
    fontFace: FONT_H, fontSize: 40, bold: true, color, margin: 0 });
  slide.addText(label, { x: x + 0.2, y: y + 1.02, w: 3.3, h: 0.6,
    fontFace: FONT_B, fontSize: 12.5, color: MUTED, margin: 0 });
}
function footer(slide, n) {
  slide.addText(`${n} / 14`, { x: 12.3, y: 7.05, w: 0.8, h: 0.35,
    fontFace: FONT_B, fontSize: 10, color: MUTED, align: "right", margin: 0 });
  slide.addText("Houssemamor - artificial-life-neuroevolution", {
    x: 0.6, y: 7.05, w: 6, h: 0.35, fontFace: FONT_B, fontSize: 10,
    color: MUTED, margin: 0 });
}

// ---------------------------------------------------------------- 1 title
{
  const s = darkSlide();
  s.addText("Artificial Life Neuroevolution", { x: 0.8, y: 2.0, w: 11.7, h: 1.3,
    fontFace: FONT_H, fontSize: 54, bold: true, color: TEXT, margin: 0 });
  s.addText("Fixed-topology neuroevolution with a fully specified NEAT extension - eight phases, one measured pipeline", {
    x: 0.8, y: 3.45, w: 11.0, h: 0.8, fontFace: FONT_B, fontSize: 19,
    color: CYAN, margin: 0 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.8, y: 4.6, w: 5.4, h: 1.0,
    fill: { color: PANEL }, line: { color: "24304A" }, rectRadius: 0.08 });
  s.addText([
    { text: "10,762", options: { fontSize: 26, bold: true, color: CYAN } },
    { text: " steps/s at 250 agents    ", options: { fontSize: 14, color: TEXT } },
    { text: "+0.41", options: { fontSize: 26, bold: true, color: GREEN } },
    { text: " fitness gain / 12 gens", options: { fontSize: 14, color: TEXT } },
  ], { x: 1.0, y: 4.85, w: 5.0, h: 0.6, fontFace: FONT_B, margin: 0 });
  s.addNotes("Project framing: a complete neuroevolution stack - sensors, batched inference, energy economy, GA/NEAT, and a confirmatory experiment program with paired statistics.");
}

// ---------------------------------------------------------------- 2 motivation
{
  const s = darkSlide();
  title(s, "Motivation", "Why evolve the controller instead of writing it?");
  bullets(s, [
    { text: "Hand-written controllers encode the designer's guesses about a world that keeps changing", bold: true },
    { text: "A neurocontroller only needs sensors and a cost function - the behavior is discovered, not specified", sub: true },
    { text: "Artificial life adds the hard parts: energy budgets, mortality, food, competition, co-evolution" },
    { text: "The research question is not 'can it learn' but 'what does it learn, and can that be measured rigorously'" },
    { text: "Everything in this project is built so the answer is measurable: calibrated fitness, frozen checkpoints, paired statistics", color: CYAN },
  ]);
  s.addShape(pres.ShapeType.ellipse, { x: 9.6, y: 2.6, w: 2.6, h: 2.6,
    fill: { color: PANEL }, line: { color: CYAN, width: 2 } });
  s.addText("7 rays\n12 inputs\n3 actions", { x: 9.6, y: 3.25, w: 2.6, h: 1.3,
    fontFace: FONT_B, fontSize: 14, color: TEXT, align: "center", margin: 0 });
  footer(s, 2);
}

// ---------------------------------------------------------------- 3 questions
{
  const s = darkSlide();
  title(s, "Research Questions");
  const cards = [
    ["Q1", "Can a small brain learn an artificial-life world under a measured, calibrated fitness?", CYAN],
    ["Q2", "Does behavior depend on the fitness weighting (Experiment D) and the crossover operator (Experiment E)?", AMBER],
    ["Q3", "Does learned behavior transfer to unseen layouts - and does it survive topology evolution (NEAT)?", GREEN],
  ];
  cards.forEach(([tag, text, color], i) => {
    const y = 2.2 + i * 1.55;
    s.addShape(pres.ShapeType.roundRect, { x: 0.8, y, w: 11.7, h: 1.3,
      fill: { color: PANEL }, line: { color: "24304A" }, rectRadius: 0.06 });
    s.addShape(pres.ShapeType.ellipse, { x: 1.05, y: y + 0.3, w: 0.7, h: 0.7,
      fill: { color }, line: { color } });
    s.addText(tag, { x: 1.05, y: y + 0.44, w: 0.7, h: 0.4, fontFace: FONT_B,
      fontSize: 15, bold: true, color: INK, align: "center", margin: 0 });
    s.addText(text, { x: 2.0, y: y + 0.25, w: 10.2, h: 0.85, fontFace: FONT_B,
      fontSize: 16, color: TEXT, valign: "middle", margin: 0 });
  });
  s.addNotes("Q1-Q3 map directly onto the experiment families: A/B (generalization), D, E, and Phase 7 (NEAT).");
  footer(s, 3);
}

// ---------------------------------------------------------------- 4 related work
{
  const s = darkSlide();
  title(s, "Related Work", "Precedented here vs. built here");
  bullets(s, [
    { text: "Sims (1994) - evolutionary rule-based creatures: precedent for evolved control in ALife", bold: true },
    { text: "Stanley & Miikkulainen (2002) - NEAT: innovation tracking, compatibility-distance speciation, fitness sharing", bold: true },
    { text: "Built here (not in prior work of this project):", color: CYAN },
    { text: "Fitness calibration before weighting - measured component scales, never hand-picked coefficients", sub: true },
    { text: "A confirmatory experiment program: seed-blocked designs, paired Wilcoxon + Holm, effect sizes + CIs", sub: true },
    { text: "Co-evolution measured against FROZEN checkpoints (hall-of-fame), not just same-generation fitness", sub: true },
    { text: "Batched inference under variable topology - the NEAT open question, resolved by measurement (2.9x)", sub: true },
  ]);
  footer(s, 4);
}

// ---------------------------------------------------------------- 5 architecture
{
  const s = darkSlide();
  title(s, "System Architecture", "One sentence per layer");
  const layers = [
    ["Sensors", "7-ray fan -> 12 observations", CYAN],
    ["Inference", "Batched (fixed) / depth-layered padded (NEAT)", GREEN],
    ["Physics + Energy", "Step-based engine, metabolic budget, mortality", AMBER],
    ["Evolution", "GA (3 crossover methods) + reproduction; full NEAT", CYAN],
    ["Analytics", "Calibration, paired stats, hall-of-fame, determinism", GREEN],
    ["Visualization", "PyGame GUI, tiered panels, inspector, web viewer", AMBER],
  ];
  layers.forEach(([name, desc, color], i) => {
    const y = 2.05 + i * 0.78;
    s.addShape(pres.ShapeType.roundRect, { x: 0.8, y, w: 3.1, h: 0.62,
      fill: { color: PANEL }, line: { color }, rectRadius: 0.05 });
    s.addText(name, { x: 0.95, y: y + 0.14, w: 2.9, h: 0.36, fontFace: FONT_B,
      fontSize: 14, bold: true, color, margin: 0 });
    s.addText(desc, { x: 4.2, y: y + 0.14, w: 8.2, h: 0.36, fontFace: FONT_B,
      fontSize: 14, color: TEXT, margin: 0 });
  });
  s.addNotes("Layered design mirrors the plan's Section 9. Each layer has its own tests; the engine contract is shared by GUI, headless runner, and all experiment runners.");
  footer(s, 5);
}

// ---------------------------------------------------------------- 6 throughput
{
  const s = darkSlide();
  title(s, "Neuroevolution Design: Throughput First",
    "One topology means one batched matmul for the whole population");
  stat(s, 0.8, 2.3, "10,762", "steps/s - batched, 250 agents", CYAN);
  stat(s, 4.8, 2.3, "19.3", "steps/s - naive per-agent", MUTED);
  stat(s, 8.8, 2.3, "557x", "speedup (358x the 30-FPS gate)", GREEN);
  bullets(s, [
    { text: "The load-bearing detail: the flat genome layout (W then b per layer) must match the batched tensor slices", bold: true },
    { text: "Without that alignment the 358x is fiction - and it silently was, until Phase 1.5 caught it" },
    { text: "NEAT breaks the trick (variable topology). Resolved by depth-layered padded batching: 249 steps/s, 2.9x per-agent (Phase 7)" },
  ], { y: 4.4, h: 2.2, w: 11.9 });
  footer(s, 6);
}

// ---------------------------------------------------------------- 7 fitness
{
  const s = darkSlide();
  title(s, "Fitness Design: Calibrate, Then Weight",
    "The project's central methodological decision");
  bullets(s, [
    { text: "Measure each component's natural scale with random-policy and stand-still probes; only then lock coefficients", bold: true },
    { text: "Survival 133.94 steps | food 1.0 item | exploration 77.76 px | collision 1.20 contacts (energy-limited dynamics)", color: CYAN },
    { text: "Why it mattered: under survival-saturated dynamics, ~90% of every fitness was the survival term - every experiment read null" },
    { text: "The fix (Phase 9): metabolic cost 0.7/step so starvation bites inside the 150-step window, food regrowth, exploration computed" },
    { text: "The nulls moved as a family - improvement signal grew 7-9x, generalization produced a consistent-direction penalty", color: GREEN, bold: true },
  ]);
  s.addShape(pres.ShapeType.roundRect, { x: 9.3, y: 5.3, w: 3.3, h: 1.2,
    fill: { color: PANEL }, line: { color: RED, width: 1.5 }, rectRadius: 0.06 });
  s.addText("A null result must be explained before it is believed", {
    x: 9.45, y: 5.55, w: 3.0, h: 0.8, fontFace: FONT_B, fontSize: 12.5,
    italic: true, color: RED, margin: 0 });
  footer(s, 7);
}

// ---------------------------------------------------------------- 8 demo
{
  const s = darkSlide();
  title(s, "Live Demo", "docs/demo_script.md - rehearsable, written against what runs");
  const steps = [
    "GUI, MVP tier: neural population, SPACE pause, F9 tiered panel",
    "GUI, advanced tier: predator/prey ecosystem - captures and births",
    "Click an agent: sensor overlay + best-agent inspector (vitals, controller graph)",
    "Control bar: Pause / x1 / x10 / Save (checkpoint) / Reset",
    "Headless determinism: same seed, byte-identical run",
    "Web viewer: uvicorn webdash.backend.main:app - compare conditions with CI + effect size",
  ];
  s.addText(steps.map((t, i) => ({
    text: t, options: { bullet: { code: "25B8", indent: 16 }, breakLine: true,
      color: TEXT, fontSize: 16.5, paraSpaceAfter: 12 } })),
    { x: 0.8, y: 2.2, w: 7.9, h: 4.4, fontFace: FONT_B, valign: "top", margin: 0 });
  s.addShape(pres.ShapeType.roundRect, { x: 9.1, y: 2.4, w: 3.4, h: 3.4,
    fill: { color: PANEL }, line: { color: CYAN, width: 1.5 }, rectRadius: 0.08 });
  s.addText("F9\nclick agent\nF for frame", { x: 9.1, y: 3.3, w: 3.4, h: 1.8,
    fontFace: FONT_B, fontSize: 17, color: CYAN, align: "center", margin: 0 });
  s.addNotes("Demo order per docs/demo_script.md; the plan's unavailable steps (checkpoint load into the GUI, video fallback) are honestly scripted in the doc.");
  footer(s, 8);
}

// ---------------------------------------------------------------- 9 generalization
{
  const s = darkSlide();
  title(s, "Results: Generalization (Experiments A/B)",
    "A consistent-direction transfer penalty - train-higher by +0.047");
  stat(s, 0.8, 2.3, "+0.047", "gap (train - unseen), 95% CI [+0.002, +0.089]", AMBER);
  stat(s, 4.8, 2.3, "p=0.106", "paired Wilcoxon, n=10 seeds", TEXT);
  stat(s, 8.8, 2.3, "r=+0.60", "matched-pairs effect, 7/10 seeds favor train", AMBER);
  bullets(s, [
    { text: "Frozen controllers transfer slightly worse to unseen layouts - the expected direction, not yet resolved at n=10" },
    { text: "History matters: 5 px dynamics gave a reverse gap (p=0.065, r=-0.67); 12 px legacy metabolic gave a clean null (p=0.85)", color: MUTED },
    { text: "The first recommendation: re-run at n=20-30. The effect is sized to resolve", color: CYAN, bold: true },
  ], { y: 4.4, h: 2.1, w: 11.9 });
  footer(s, 9);
}

// ---------------------------------------------------------------- 10 D/E
{
  const s = darkSlide();
  title(s, "Results: Weighting & Crossover (D, E)",
    "A live landscape with a null verdict: the status-quo weights win");
  stat(s, 0.8, 2.3, "+0.07-0.12", "EQUAL-weighted lead over every specialization", GREEN);
  stat(s, 4.8, 2.3, "p=0.92", "blend vs uniform: indistinguishable (r=+0.06)", TEXT);
  stat(s, 8.8, 2.3, "+0.50", "fitness gain, blend, 12 generations", CYAN);
  bullets(s, [
    { text: "Under survival-saturated dynamics all four weightings sat within 0.015 of each other by construction" },
    { text: "Now the conditions separate - and the separation favors equal weighting: down-weighting food (survival) costs the most on a replenished landscape", bold: true },
    { text: "Every comparison shows p-value + effect size + interval together - significance is never quoted alone", color: MUTED },
  ], { y: 4.4, h: 2.1, w: 11.9 });
  footer(s, 10);
}

// ---------------------------------------------------------------- 11 ecosystem
{
  const s = darkSlide();
  title(s, "Results: Predator/Prey & Hall of Fame",
    "An oscillating arms race, explained rather than hidden");
  stat(s, 0.8, 2.3, "5 -> 22", "captures per day in a viable run (seed 4)", AMBER);
  stat(s, 4.8, 2.3, "7 of 10", "random seeds collapse (predator extinction)", TEXT);
  stat(s, 8.8, 2.3, "13-28 / 6", "prey / predators at carrying capacity", GREEN);
  bullets(s, [
    { text: "Within-generation fitness in a co-evolution is relative - the hall-of-fame win rate against FROZEN archives is the absolute measure", bold: true },
    { text: "The first stability improvement was a bug: newborns silently ran the default 0.1 energy equation, not the parent's 0.25. Fixed, re-measured, and the collapse rate rose - the honest number", color: MUTED },
  ], { y: 4.4, h: 2.1, w: 11.9 });
  footer(s, 11);
}

// ---------------------------------------------------------------- 12 NEAT
{
  const s = darkSlide();
  title(s, "NEAT Extension: Topology Evolves",
    "The plan's open batching question, resolved by measurement");
  stat(s, 0.8, 2.3, "51 -> 102", "mean complexity over 30 generations", CYAN);
  stat(s, 4.8, 2.3, "2.9x", "batched vs per-agent (249 vs 85 steps/s)", GREEN);
  stat(s, 8.8, 2.3, "571", "innovations created, full Appendix C spec", AMBER);
  bullets(s, [
    { text: "Literature-accurate: innovation reuse, aligned crossover, compatibility distance (c1=1, c2=1, c3=0.4, threshold 3.0), fitness sharing, stagnation removal" },
    { text: "The honest negative: the evolutionary run keeps ONE species. The mechanism is unit-proven; the bimodal landscape (foragers vs starvers) doesn't split lineages in 30 generations", color: RED },
  ], { y: 4.4, h: 2.1, w: 11.9 });
  footer(s, 12);
}

// ---------------------------------------------------------------- 13 limitations
{
  const s = darkSlide();
  title(s, "Limitations & Future Work", "One shape, honestly");
  const rows = [
    ["Statistical power", "The best effects sit near but not past Holm at n=10. Re-run at n=20-30 is the top action.", AMBER],
    ["Bimodal landscape", "Enough leverage for the GA, not for speciation. Graded food availability is the route.", CYAN],
    ["Fragile predators", "7 of 10 ecosystem seeds collapse after the newborn-energy fix; the 0.25/step ecosystem clock is the lever.", RED],
    ["Metabolic cost is a factor", "0.7/step sets the survival-vs-forging split every result depends on - sweep 0.4/0.7/1.0.", GREEN],
  ];
  rows.forEach(([head, body, color], i) => {
    const y = 2.2 + i * 1.15;
    s.addShape(pres.ShapeType.roundRect, { x: 0.8, y, w: 2.9, h: 0.95,
      fill: { color: PANEL }, line: { color }, rectRadius: 0.05 });
    s.addText(head, { x: 0.95, y: y + 0.28, w: 2.65, h: 0.45, fontFace: FONT_B,
      fontSize: 13.5, bold: true, color, margin: 0 });
    s.addText(body, { x: 4.0, y: y + 0.2, w: 8.5, h: 0.65, fontFace: FONT_B,
      fontSize: 13.5, color: TEXT, valign: "middle", margin: 0 });
  });
  footer(s, 13);
}

// ---------------------------------------------------------------- 14 close
{
  const s = darkSlide();
  s.addText("Artifacts, all reproducible", { x: 0.8, y: 1.4, w: 11.7, h: 0.9,
    fontFace: FONT_H, fontSize: 40, bold: true, color: TEXT, margin: 0 });
  s.addText([
    { text: "248 tests green (3 Python versions x 3 OS in CI)   ", options: { color: GREEN, fontSize: 15 } },
    { text: "every number on these slides traces to a committed JSON or SVG", options: { color: MUTED, fontSize: 15 } },
  ], { x: 0.8, y: 2.4, w: 11.7, h: 0.5, fontFace: FONT_B, margin: 0 });
  s.addText([
    { text: "docs/final_report.md", options: { bullet: { code: "25AA" }, bold: true, color: CYAN, breakLine: true } },
    { text: "experiments/EXP-*/ - frozen run outputs (stats summaries, SVG plots, stability sweep)", options: { bullet: { code: "25AA" }, breakLine: true } },
    { text: "webdash/ - read-only FastAPI + Bootstrap viewer (uvicorn webdash.backend.main:app)", options: { bullet: { code: "25AA" }, breakLine: true } },
    { text: "PLAN.md - the eight phases and their exit criteria, with the honest exceptions recorded", options: { bullet: { code: "25AA" }, breakLine: true } },
  ], { x: 0.8, y: 3.3, w: 11.5, h: 2.4, fontFace: FONT_B, fontSize: 16,
    color: TEXT, valign: "top", margin: 0 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.8, y: 6.0, w: 11.7, h: 0.85,
    fill: { color: PANEL }, line: { color: "24304A" }, rectRadius: 0.06 });
  s.addText("A null result must be explained before it is believed. This project ran its nulls down to the dynamics and fixed the dynamics.",
    { x: 1.0, y: 6.22, w: 11.3, h: 0.45, fontFace: FONT_H, fontSize: 15,
      italic: true, color: CYAN, margin: 0 });
  s.addNotes("Close on the method, not just the numbers: the discipline of explaining nulls is what produced the project's strongest result.");
}

pres.writeFile({ fileName: "docs/presentation_deck.pptx" })
  .then(() => console.log("wrote docs/presentation_deck.pptx"));
