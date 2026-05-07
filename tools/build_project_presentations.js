const fs = require("fs");
const path = require("path");
const PptxGenJS = require("C:/Users/singh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs");

const ROOT = "C:/CSCI165";
const COLORS = {
  bg: "F8F9FB",
  navy: "1B2B41",
  blue: "3B5DC9",
  teal: "228B9A",
  gray: "5C6370",
  light: "E4E9F2",
  white: "FFFFFF",
  alt: "F2F5FA",
};

function readCsv(csvPath) {
  const raw = fs.readFileSync(csvPath, "utf8").trim().split(/\r?\n/);
  const headers = raw[0].split(",");
  return raw.slice(1).map((line) => {
    const cells = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = cells[i] ?? "";
    });
    return row;
  });
}

function newDeck(subject) {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "OpenAI Codex";
  pptx.company = "OpenAI";
  pptx.subject = subject;
  pptx.title = subject;
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Aptos Display",
    bodyFontFace: "Aptos",
    lang: "en-US",
  };
  return pptx;
}

function setBg(slide) {
  slide.background = { color: COLORS.bg };
}

function addHeader(slide, title, subtitle = "") {
  slide.addText(title, {
    x: 0.55, y: 0.32, w: 8.5, h: 0.38,
    fontFace: "Aptos Display", fontSize: 24, bold: true, color: COLORS.navy,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.58, y: 0.82, w: 10.6, h: 0.2,
      fontFace: "Aptos", fontSize: 10.5, color: COLORS.gray, margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.55, y: 1.16, w: 2.3, h: 0.05,
    line: { color: COLORS.blue, transparency: 100 },
    fill: { color: COLORS.blue },
  });
}

function addCard(slide, x, y, w, h) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.06,
    line: { color: COLORS.light, pt: 1 },
    fill: { color: COLORS.white },
  });
}

function addTitleSlide(pptx, title, subtitle, courseLine) {
  const slide = pptx.addSlide();
  setBg(slide);
  addCard(slide, 0.75, 0.9, 11.2, 4.7);
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.75, y: 0.9, w: 0.22, h: 4.7,
    line: { color: COLORS.blue, transparency: 100 },
    fill: { color: COLORS.blue },
  });
  slide.addText(title, {
    x: 1.25, y: 1.45, w: 9.9, h: 1.2,
    fontFace: "Aptos Display", fontSize: 28, bold: true, color: COLORS.navy, margin: 0,
  });
  slide.addText(subtitle, {
    x: 1.28, y: 3.02, w: 8.8, h: 0.45,
    fontFace: "Aptos", fontSize: 18, color: COLORS.teal, margin: 0,
  });
  slide.addText(courseLine, {
    x: 1.28, y: 4.1, w: 8.8, h: 0.35,
    fontFace: "Aptos", fontSize: 14, color: COLORS.gray, margin: 0,
  });
}

function addBulletList(slide, bullets, x, y, w, h, size = 19) {
  const runs = [];
  bullets.forEach((bullet, i) => {
    runs.push({
      text: bullet,
      options: { bullet: { indent: 14 }, breakLine: i !== bullets.length - 1 },
    });
  });
  slide.addText(runs, {
    x, y, w, h,
    fontFace: "Aptos", fontSize: size, color: COLORS.navy,
    paraSpaceAfterPt: 10, valign: "top", breakLine: false,
    margin: 0.06,
  });
}

function addBulletsSlide(pptx, title, bullets, subtitle = "") {
  const slide = pptx.addSlide();
  setBg(slide);
  addHeader(slide, title, subtitle);
  addCard(slide, 0.7, 1.55, 11.0, 5.45);
  addBulletList(slide, bullets, 1.0, 1.92, 10.2, 4.8, 20);
}

function addTwoColumnSlide(pptx, title, leftTitle, leftBullets, rightTitle, rightBullets, subtitle = "") {
  const slide = pptx.addSlide();
  setBg(slide);
  addHeader(slide, title, subtitle);
  addCard(slide, 0.65, 1.55, 5.35, 5.45);
  addCard(slide, 6.1, 1.55, 5.25, 5.45);

  slide.addText(leftTitle, {
    x: 0.95, y: 1.84, w: 4.4, h: 0.28,
    fontFace: "Aptos Display", fontSize: 18, bold: true, color: COLORS.blue, margin: 0,
  });
  slide.addText(rightTitle, {
    x: 6.38, y: 1.84, w: 4.5, h: 0.28,
    fontFace: "Aptos Display", fontSize: 18, bold: true, color: COLORS.blue, margin: 0,
  });
  addBulletList(slide, leftBullets, 0.92, 2.28, 4.75, 4.35, 17);
  addBulletList(slide, rightBullets, 6.34, 2.28, 4.6, 4.35, 17);
}

function addTableSlide(pptx, title, headers, rows, subtitle = "") {
  const slide = pptx.addSlide();
  setBg(slide);
  addHeader(slide, title, subtitle);

  const tableRows = [
    headers.map((h) => ({
      text: h,
      options: {
        bold: true,
        color: COLORS.white,
        fill: COLORS.blue,
        align: "center",
        valign: "mid",
        fontSize: 12,
      },
    })),
    ...rows.map((row, rowIdx) => row.map((cell) => ({
      text: cell,
      options: {
        color: COLORS.navy,
        fill: rowIdx % 2 === 0 ? COLORS.white : COLORS.alt,
        align: "center",
        valign: "mid",
        fontSize: 11,
      },
    }))),
  ];
  slide.addTable(tableRows, {
    x: 0.62, y: 1.62, w: 11.1, h: 5.15,
    border: { type: "solid", pt: 1, color: COLORS.light },
    fill: COLORS.white,
    color: COLORS.navy,
    fontFace: "Aptos",
    fontSize: 11,
    align: "center",
    valign: "mid",
    rowH: 0.48,
    autoFit: false,
    margin: 0.04,
    bold: false,
  });
}

function addOneImageSlide(pptx, title, imagePath, caption, subtitle = "") {
  const slide = pptx.addSlide();
  setBg(slide);
  addHeader(slide, title, subtitle);
  addCard(slide, 0.75, 1.55, 10.9, 5.45);
  slide.addImage({ path: imagePath, x: 1.0, y: 1.85, w: 10.4, h: 4.45 });
  slide.addText(caption, {
    x: 1.0, y: 6.42, w: 10.3, h: 0.2,
    fontFace: "Aptos", fontSize: 11, color: COLORS.gray, align: "center", margin: 0,
  });
}

function addTwoImageSlide(pptx, title, leftImage, leftCaption, rightImage, rightCaption, subtitle = "") {
  const slide = pptx.addSlide();
  setBg(slide);
  addHeader(slide, title, subtitle);
  addCard(slide, 0.7, 1.6, 5.0, 4.95);
  addCard(slide, 6.2, 1.6, 5.0, 4.95);
  slide.addImage({ path: leftImage, x: 0.86, y: 1.82, w: 4.68, h: 3.95 });
  slide.addImage({ path: rightImage, x: 6.36, y: 1.82, w: 4.68, h: 3.95 });
  slide.addText(leftCaption, {
    x: 0.82, y: 5.92, w: 4.76, h: 0.36,
    fontFace: "Aptos", fontSize: 10.5, color: COLORS.gray, align: "center", margin: 0,
  });
  slide.addText(rightCaption, {
    x: 6.32, y: 5.92, w: 4.76, h: 0.36,
    fontFace: "Aptos", fontSize: 10.5, color: COLORS.gray, align: "center", margin: 0,
  });
}

function build8Queens() {
  const project = path.join(ROOT, "8Queens");
  const pptx = newDeck("8-Queens project presentation");
  const summary = readCsv(path.join(project, "results", "summary.csv"));

  addTitleSlide(
    pptx,
    "Incremental Evolutionary Algorithm for 8-Queens",
    "Motivation, method, evaluation, and results",
    "CSCI 165 project presentation"
  );

  addTwoColumnSlide(
    pptx,
    "Motivation And Problem Statement",
    "Why This Domain Matters",
    [
      "8-Queens is a classic constraint satisfaction and search problem.",
      "It is easy to visualize, but still useful for comparing optimization strategies.",
      "It shows how a smart representation can shrink the search space dramatically."
    ],
    "Problem Addressed",
    [
      "Place 8 queens on a chessboard so that no two attack each other.",
      "Compare an evolutionary algorithm under different population, mutation, and tournament settings.",
      "Measure success rate, solution quality, runtime, and generations needed to solve."
    ]
  );

  addTwoColumnSlide(
    pptx,
    "Background And Approach",
    "Background",
    [
      "A valid solution has no shared rows and no diagonal conflicts.",
      "The project uses a permutation encoding, so each column has one queen and rows do not repeat.",
      "Fitness = 28 minus the number of attacking pairs, so 28 is perfect."
    ],
    "Method",
    [
      "Initialize a population of random row permutations.",
      "Use tournament selection, cut-and-crossfill crossover, and swap mutation.",
      "Keep elite individuals and iterate until a perfect board is found or the generation budget ends."
    ]
  );

  addBulletsSlide(pptx, "Experiment Setup", [
    "30 runs per configuration with max_gens = 1000 and elitism = 2.",
    "Population sweep: 20, 50, 100 with mutation = 0.10 and tournament = 3.",
    "Mutation sweep: 0.05, 0.10, 0.20 with population = 100 and tournament = 3.",
    "Tournament sweep: 2, 3, 5 with population = 100 and mutation = 0.10.",
    "Metrics: success rate, mean best fitness, average generation solved, and runtime."
  ]);

  addTableSlide(
    pptx,
    "Evaluation Results",
    ["Config", "Success", "Mean Fitness", "Avg Gen Solved", "Time (ms)"],
    summary.map((r) => [
      r.config,
      r.success_rate,
      r.mean_fitness,
      r.avg_gen_solved,
      r.mean_time_ms,
    ]),
    "Summary across 30 runs per configuration"
  );

  addTwoImageSlide(
    pptx,
    "Visual Results",
    path.join(project, "figures", "success_rates.png"),
    "Success rates show that `pop=20` was the only configuration with notable failures.",
    path.join(project, "figures", "convergence_plot.png"),
    "Convergence curves show the stronger settings reaching perfect fitness quickly."
  );

  addTwoImageSlide(
    pptx,
    "Representation And Example Boards",
    path.join(project, "figures", "search_space.png"),
    "Permutation encoding reduces the search space from billions of boards to 40,320 candidates.",
    path.join(project, "figures", "example_boards.png"),
    "The board visualization makes conflicts and solved states easy to explain."
  );

  addBulletsSlide(pptx, "Conclusions And Future Work", [
    "The permutation encoding was the key design choice because it removed many invalid boards before search even began.",
    "Most tested settings solved the problem perfectly in all 30 runs; `pop=20` was the weakest at 23/30 successes.",
    "Population sizes of 50 and 100 provided perfect success with fast convergence, while higher tournament pressure slowed solving.",
    "Future work: test larger N-Queens variants, compare against hill climbing or simulated annealing, and vary crossover and elitism."
  ]);

  const out = path.join(project, "8Queens_project_presentation.pptx");
  return pptx.writeFile({ fileName: out }).then(() => out);
}

function buildRastrigin() {
  const project = path.join(ROOT, "Rastrigin");
  const pptx = newDeck("Rastrigin project presentation");
  const summary = readCsv(path.join(project, "results", "summary.csv"));
  const byName = Object.fromEntries(summary.map((row) => [row.algorithm, row]));

  addTitleSlide(
    pptx,
    "Gradient Descent vs Simulated Annealing on Rastrigin",
    "Comparing local optimization and stochastic search on a multimodal landscape",
    "CSCI 165 project presentation"
  );

  addTwoColumnSlide(
    pptx,
    "Motivation And Problem Statement",
    "Why This Domain Matters",
    [
      "Optimization appears in machine learning, engineering design, planning, and scientific computing.",
      "Rastrigin is a standard benchmark because it has many local minima but a known global optimum.",
      "It is a good test of whether a method exploits quickly or explores broadly."
    ],
    "Problem Addressed",
    [
      "Minimize the 2D Rastrigin function over the bounded domain [-5.12, 5.12].",
      "Compare three gradient descent variants against simulated annealing.",
      "Evaluate whether each method can reliably reach or closely approach the global minimum f(0,0) = 0."
    ]
  );

  addTwoColumnSlide(
    pptx,
    "Background And Approach",
    "Background",
    [
      "Rastrigin combines a quadratic bowl with cosine ripples, creating many deceptive local minima.",
      "Gradient descent follows local slope information and can converge quickly with a suitable step schedule.",
      "Simulated annealing accepts occasional worse moves early, which can help it escape local traps."
    ],
    "Methods Used",
    [
      "GD Fixed: constant learning rate alpha = 0.01.",
      "GD Decaying: alpha0 = 0.1 with decay = 0.001.",
      "GD Momentum: alpha = 0.01 and beta = 0.9.",
      "SA: T0 = 10, alpha = 0.995, step radius = 0.5."
    ]
  );

  addBulletsSlide(pptx, "Experiment Setup", [
    "30 runs per algorithm from shared random starting points across the full domain.",
    "Maximum of 5000 iterations for every method.",
    "Success threshold: f(x) < 1e-3 counts as reaching the global minimum.",
    "Metrics: best objective value, success rate, distance to optimum, and runtime.",
    "Using the same starting points across algorithms kept the comparison fair."
  ]);

  addTableSlide(
    pptx,
    "Evaluation Results",
    ["Algorithm", "Mean f", "Best f", "Success", "Time (ms)"],
    ["GD_Fixed", "GD_Decaying", "GD_Momentum", "SA"].map((name) => [
      name,
      byName[name].mean_f,
      byName[name].best_f,
      byName[name].success_rate,
      byName[name].mean_time_ms,
    ]),
    "Summary across 30 repeated trials"
  );

  addTwoImageSlide(
    pptx,
    "Landscape And Search Behavior",
    path.join(project, "figures", "rastrigin_contour.png"),
    "The contour plot shows the highly multimodal landscape around the global minimum.",
    path.join(project, "figures", "trajectories.png"),
    "Trajectory plots show the contrast between direct descent and exploratory stochastic search."
  );

  addTwoImageSlide(
    pptx,
    "Convergence And Result Distribution",
    path.join(project, "figures", "convergence_plot.png"),
    "Decaying GD converged consistently to near-zero values across all runs.",
    path.join(project, "figures", "boxplot.png"),
    "Fixed GD and momentum were less reliable, while SA often got close without crossing the strict success threshold."
  );

  addBulletsSlide(pptx, "Conclusions And Future Work", [
    "The best overall performer was decaying gradient descent, which achieved 30/30 successes with mean f approximately 1e-6.",
    "Simulated annealing improved exploration and found near-optimal points, but under this schedule it did not satisfy the strict success threshold in any run.",
    "Fixed learning rate and momentum were vulnerable to local minima despite reasonable runtimes.",
    "Future work: tune the annealing schedule, test additional benchmark functions, and examine higher-dimensional Rastrigin settings."
  ]);

  const out = path.join(project, "Rastrigin_project_presentation.pptx");
  return pptx.writeFile({ fileName: out }).then(() => out);
}

function buildTSP() {
  const project = path.join(ROOT, "TSP");
  const pptx = newDeck("TSP project presentation");
  const summary = readCsv(path.join(project, "results", "summary.csv"));
  const byName = Object.fromEntries(summary.map((row) => [row.algorithm, row]));

  addTitleSlide(
    pptx,
    "TSP With Hill Climbing, Simulated Annealing, And Threshold Accepting",
    "Comparing local-search methods on a fixed 100-city traveling salesman instance",
    "CSCI 165 project presentation"
  );

  addTwoColumnSlide(
    pptx,
    "Motivation And Problem Statement",
    "Why This Domain Matters",
    [
      "The Traveling Salesman Problem is a classic optimization problem with real scheduling and routing relevance.",
      "It appears in logistics, circuit design, delivery planning, and manufacturing.",
      "It is computationally hard, so heuristic and local-search methods are often used in practice."
    ],
    "Problem Addressed",
    [
      "Find a short closed tour through 100 cities and return to the starting city.",
      "Compare hill climbing, simulated annealing, and threshold accepting under a shared evaluation budget.",
      "Determine which method finds the shortest tours most consistently on the same city instance."
    ]
  );

  addTwoColumnSlide(
    pptx,
    "Background And Approach",
    "Background",
    [
      "A TSP solution is a permutation of city indices representing visit order.",
      "The objective is to minimize total tour length, including the return to the first city.",
      "Local search for TSP often relies on neighborhood operators such as swap and 2-opt reversal."
    ],
    "Methods Used",
    [
      "Hill Climbing: accepts only improving moves, with both plain and restart variants.",
      "Simulated Annealing: sometimes accepts worse moves with probability based on temperature.",
      "Threshold Accepting: deterministically accepts moves within a shrinking threshold.",
      "All methods used the 2-opt neighborhood and a budget of 100,000 evaluations."
    ]
  );

  addBulletsSlide(pptx, "Experiment Setup", [
    "Fixed 100-city dataset reused across all algorithms for fairness.",
    "30 repeated runs per algorithm with the same total evaluation budget.",
    "Algorithms compared: HC_plain, HC_restart, SA_fast, SA_slow, and TA.",
    "Metrics: mean tour cost, standard deviation, best and worst tour cost, and runtime.",
    "The main comparison question was which method best balances exploration and solution quality."
  ]);

  addTableSlide(
    pptx,
    "Evaluation Results",
    ["Algorithm", "Mean Cost", "Best Cost", "Std Dev", "Time (ms)"],
    ["HC_plain", "HC_restart", "SA_fast", "SA_slow", "TA"].map((name) => [
      name,
      byName[name].mean_cost,
      byName[name].best_cost,
      byName[name].std_cost,
      byName[name].mean_time_ms,
    ]),
    "Summary across 30 runs per algorithm"
  );

  addTwoImageSlide(
    pptx,
    "City Instance And Best Routes",
    path.join(project, "figures", "city_map.png"),
    "All algorithms were tested on the same fixed 100-city dataset.",
    path.join(project, "figures", "best_routes.png"),
    "The route visualizations show how different search strategies shape the final tour."
  );

  addTwoImageSlide(
    pptx,
    "Convergence And Cooling Behavior",
    path.join(project, "figures", "convergence_plot.png"),
    "SA_slow converged to the strongest average solution quality over the run set.",
    path.join(project, "figures", "temperature_schedule.png"),
    "Cooling rate mattered: slower annealing preserved exploration longer and improved solution quality."
  );

  addBulletsSlide(pptx, "Conclusions And Future Work", [
    "The best-performing method was SA_slow, with mean cost 815.33 and best cost 781.98 across the 30 runs.",
    "Hill climbing was competitive and improved further with restarts, but it remained more vulnerable to local minima.",
    "SA_fast cooled too quickly and performed worst on average, showing that exploration needs enough time to be useful.",
    "Threshold accepting performed between hill climbing and the weaker annealing schedule, offering a deterministic compromise.",
    "Future work: test larger city sets, tune restart counts and thresholds, and compare against genetic algorithms or ant-colony methods."
  ]);

  const out = path.join(project, "TSP_project_presentation.pptx");
  return pptx.writeFile({ fileName: out }).then(() => out);
}

const pptx = new PptxGenJS();

async function main() {
  const outs = [];
  outs.push(await build8Queens());
  outs.push(await buildTSP());
  outs.push(await buildRastrigin());
  console.log(outs.join("\n"));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
