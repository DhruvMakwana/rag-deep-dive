/*
 * Reusable scenario-based QnA widget. Drop a container into any page:
 *
 *   <div class="quiz-widget" data-title="Scenario Check: My Topic">
 *   <script type="application/json">
 *   { "questions": [ { "scenario": "...", "question": "...",
 *       "options": ["...", "...", "...", "..."], "correct": 1,
 *       "explanations": ["why A", "why B", "why C", "why D"] } ] }
 *   </script>
 *   </div>
 *
 * Client-side only — nothing is stored or sent anywhere. Score resets on
 * reload/restart by design (see the blog page for why: no accounts, no
 * tracking, just an honest self-check while reading).
 */
(function () {
  function renderQuiz(container) {
    if (container.dataset.quizInitialized) return;

    const dataScript = container.querySelector('script[type="application/json"]');
    if (!dataScript) return;

    let data;
    try {
      data = JSON.parse(dataScript.textContent);
    } catch (e) {
      console.error("quiz.js: failed to parse question data", e);
      return;
    }

    const questions = data.questions || [];
    if (!questions.length) return;

    container.dataset.quizInitialized = "true";
    const title = container.getAttribute("data-title") || "Scenario Check";

    let current = 0;
    const answered = new Array(questions.length).fill(null);

    container.innerHTML = "";
    container.classList.add("quiz-widget-active");

    const header = document.createElement("div");
    header.className = "quiz-header";
    header.innerHTML =
      '<span class="quiz-title">' + escapeHtml(title) + '</span>' +
      '<span class="quiz-score">Score: <strong><span class="quiz-score-num">0</span>/' + questions.length + '</strong></span>';
    container.appendChild(header);

    const progress = document.createElement("div");
    progress.className = "quiz-progress";
    container.appendChild(progress);

    const body = document.createElement("div");
    body.className = "quiz-body";
    container.appendChild(body);

    const nav = document.createElement("div");
    nav.className = "quiz-nav";
    const prevBtn = mkBtn("← Prev", "quiz-btn quiz-btn-ghost");
    const restartBtn = mkBtn("Restart", "quiz-btn quiz-btn-ghost");
    const nextBtn = mkBtn("Next →", "quiz-btn quiz-btn-primary");
    nav.append(prevBtn, restartBtn, nextBtn);
    container.appendChild(nav);

    function mkBtn(label, cls) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.className = cls;
      return b;
    }

    function scoreNum() {
      return answered.filter((a, i) => a !== null && a === questions[i].correct).length;
    }

    function renderQuestion() {
      const q = questions[current];
      progress.textContent = "Question " + (current + 1) + " of " + questions.length;
      header.querySelector(".quiz-score-num").textContent = scoreNum();

      body.innerHTML = "";

      if (q.source) {
        const src = document.createElement("div");
        src.className = "quiz-source";
        src.textContent = "From: " + q.source;
        body.appendChild(src);
      }

      if (q.scenario) {
        const scenario = document.createElement("p");
        scenario.className = "quiz-scenario";
        scenario.textContent = q.scenario;
        body.appendChild(scenario);
      }

      const qEl = document.createElement("p");
      qEl.className = "quiz-question";
      qEl.textContent = q.question;
      body.appendChild(qEl);

      const optsEl = document.createElement("div");
      optsEl.className = "quiz-options";
      const chosen = answered[current];

      q.options.forEach((opt, i) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "quiz-option";
        btn.textContent = opt;
        if (chosen !== null) {
          btn.disabled = true;
          if (i === q.correct) btn.classList.add("quiz-correct");
          else if (i === chosen) btn.classList.add("quiz-incorrect");
        }
        btn.addEventListener("click", function () {
          if (answered[current] !== null) return;
          answered[current] = i;
          renderQuestion();
        });
        optsEl.appendChild(btn);
      });
      body.appendChild(optsEl);

      if (chosen !== null) {
        const exp = document.createElement("div");
        const isCorrect = chosen === q.correct;
        exp.className = "quiz-explanation " + (isCorrect ? "quiz-explanation-correct" : "quiz-explanation-incorrect");
        const label = isCorrect ? "Correct — " : "Not quite — ";
        const text = (q.explanations && q.explanations[chosen]) || "";
        exp.textContent = label + text;
        body.appendChild(exp);
      }

      prevBtn.disabled = current === 0;
      nextBtn.disabled = current === questions.length - 1;
    }

    prevBtn.addEventListener("click", function () {
      if (current > 0) { current--; renderQuestion(); }
    });
    nextBtn.addEventListener("click", function () {
      if (current < questions.length - 1) { current++; renderQuestion(); }
    });
    restartBtn.addEventListener("click", function () {
      answered.fill(null);
      current = 0;
      renderQuestion();
    });

    renderQuestion();
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function initAll() {
    document.querySelectorAll(".quiz-widget").forEach(renderQuiz);
  }

  // Material's instant-navigation swaps page content via fetch; document$
  // fires on every such swap (including the first load), so subscribing
  // here is the correct place to (re)initialize widgets on each page.
  if (typeof document$ !== "undefined") {
    document$.subscribe(initAll);
  } else {
    document.addEventListener("DOMContentLoaded", initAll);
  }
})();
