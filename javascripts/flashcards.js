/*
 * Reusable flip-card widget. Drop a container into any page:
 *
 *   <div class="flashcard-widget" data-title="Flashcards: My Topic">
 *   <script type="application/json">
 *   { "cards": [ { "front": "...", "back": "...", "source": "..." } ] }
 *   </script>
 *   </div>
 *
 * `source` is optional (used on the combined Revision page to show which
 * page a card came from; per-page decks omit it). Client-side only —
 * nothing is stored or sent anywhere, shuffle/progress resets on reload.
 */
(function () {
  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function renderDeck(container) {
    if (container.dataset.flashcardsInitialized) return;

    const dataScript = container.querySelector('script[type="application/json"]');
    if (!dataScript) return;

    let data;
    try {
      data = JSON.parse(dataScript.textContent);
    } catch (e) {
      console.error("flashcards.js: failed to parse card data", e);
      return;
    }

    const originalCards = data.cards || [];
    if (!originalCards.length) return;

    container.dataset.flashcardsInitialized = "true";
    const title = container.getAttribute("data-title") || "Flashcards";

    let order = originalCards.map((_, i) => i);
    let current = 0;
    let flipped = false;

    container.innerHTML = "";
    container.classList.add("flashcard-widget-active");

    const header = document.createElement("div");
    header.className = "flashcard-header";
    header.innerHTML =
      '<span class="flashcard-title">' + escapeHtml(title) + '</span>' +
      '<span class="flashcard-progress">Card <span class="flashcard-progress-num">1</span>/' + originalCards.length + '</span>';
    container.appendChild(header);

    const cardEl = document.createElement("div");
    cardEl.className = "flashcard";
    container.appendChild(cardEl);

    const nav = document.createElement("div");
    nav.className = "flashcard-nav";
    const prevBtn = mkBtn("← Prev", "flashcard-btn flashcard-btn-ghost");
    const shuffleBtn = mkBtn("Shuffle", "flashcard-btn flashcard-btn-ghost");
    const nextBtn = mkBtn("Next →", "flashcard-btn flashcard-btn-primary");
    nav.append(prevBtn, shuffleBtn, nextBtn);
    container.appendChild(nav);

    const hint = document.createElement("div");
    hint.className = "flashcard-hint";
    hint.textContent = "Click the card to flip it";
    container.appendChild(hint);

    function mkBtn(label, cls) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.className = cls;
      return b;
    }

    function renderCard() {
      const idx = order[current];
      const card = originalCards[idx];
      flipped = false;

      header.querySelector(".flashcard-progress-num").textContent = current + 1;

      cardEl.innerHTML = "";
      cardEl.classList.remove("flashcard-flipped");

      if (card.source) {
        const src = document.createElement("div");
        src.className = "flashcard-source";
        src.textContent = "From: " + card.source;
        cardEl.appendChild(src);
      }

      const face = document.createElement("div");
      face.className = "flashcard-face";
      face.textContent = card.front;
      cardEl.appendChild(face);

      prevBtn.disabled = current === 0;
      nextBtn.disabled = current === order.length - 1;
    }

    cardEl.addEventListener("click", function () {
      const idx = order[current];
      const card = originalCards[idx];
      flipped = !flipped;
      cardEl.classList.toggle("flashcard-flipped", flipped);
      const face = cardEl.querySelector(".flashcard-face");
      if (face) face.textContent = flipped ? card.back : card.front;
    });

    prevBtn.addEventListener("click", function () {
      if (current > 0) { current--; renderCard(); }
    });
    nextBtn.addEventListener("click", function () {
      if (current < order.length - 1) { current++; renderCard(); }
    });
    shuffleBtn.addEventListener("click", function () {
      order = shuffle(originalCards.map((_, i) => i));
      current = 0;
      renderCard();
    });

    renderCard();
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function initAll() {
    document.querySelectorAll(".flashcard-widget").forEach(renderDeck);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(initAll);
  } else {
    document.addEventListener("DOMContentLoaded", initAll);
  }
})();
