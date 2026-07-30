/**
 * UI panel rendering functions.
 * Each function updates specific DOM elements by ID/class.
 */

// ── Match Card ──────────────────────────────────────────────

export function updateMatch(meme, score) {
  const img = document.getElementById('match-image');
  const empty = document.getElementById('match-empty');
  const name = document.getElementById('match-name');
  const fill = document.getElementById('match-score-fill');
  const label = document.getElementById('match-score-label');

  if (!meme) {
    clearMatch();
    return;
  }

  if (img) {
    img.src = `/memes/${meme.filename}`;
    img.alt = meme.name;
    img.hidden = false;
  }
  if (empty) empty.hidden = true;
  if (name) name.textContent = meme.name;
  if (fill) fill.style.width = `${Math.min(score, 100)}%`;
  if (label) label.textContent = `${Math.round(score)}%`;
}

export function clearMatch() {
  const img = document.getElementById('match-image');
  const empty = document.getElementById('match-empty');
  const name = document.getElementById('match-name');
  const fill = document.getElementById('match-score-fill');
  const label = document.getElementById('match-score-label');

  if (img) { img.src = ''; img.hidden = true; }
  if (empty) empty.hidden = false;
  if (name) name.textContent = 'Waiting…';
  if (fill) fill.style.width = '0%';
  if (label) label.textContent = '0%';
}

// ── Status Indicators ───────────────────────────────────────

export function updateFps(fps) {
  const el = document.getElementById('status-fps');
  if (el) el.textContent = `FPS: ${Math.round(fps)}`;
}

export function updateCameraState(active) {
  const el = document.getElementById('status-camera');
  if (el) {
    el.textContent = active ? 'Camera: ON' : 'Camera: OFF';
    el.classList.toggle('status-pill--error', !active);
  }
  const placeholder = document.getElementById('video-placeholder');
  if (placeholder) {
    placeholder.hidden = active;
  }
}

export function updateMemeCount(count) {
  const el = document.getElementById('status-memes');
  if (el) el.textContent = `Memes: ${count}`;
}

// ── Meme Gallery Strip ──────────────────────────────────────

let galleryBuilt = false;

export function updateGallery(memes, activeMeme) {
  const gallery = document.getElementById('gallery');
  if (!gallery) return;

  // Build gallery thumbnails once
  if (!galleryBuilt) {
    gallery.innerHTML = '';
    for (const meme of memes) {
      const img = document.createElement('img');
      img.src = `/memes/${meme.filename}`;
      img.alt = meme.name;
      img.className = 'gallery__item';
      img.dataset.filename = meme.filename;
      img.loading = 'lazy';
      gallery.appendChild(img);
    }
    galleryBuilt = true;
  }

  // Highlight active meme
  const items = gallery.querySelectorAll('.gallery__item');
  for (const item of items) {
    if (activeMeme && item.dataset.filename === activeMeme.filename) {
      item.classList.add('gallery__item--active');
    } else {
      item.classList.remove('gallery__item--active');
    }
  }
}

// ── Expression History Sparkline ────────────────────────────

const scores = [];
const MAX_HISTORY = 50;

export function updateHistory(score) {
  const line = document.getElementById('history-line');
  if (!line) return;

  scores.push(score);
  if (scores.length > MAX_HISTORY) scores.shift();

  const svgWidth = 300;
  const svgHeight = 60;

  const points = scores
    .map((s, i) => {
      const x = (i / (MAX_HISTORY - 1)) * svgWidth;
      const y = svgHeight - (Math.min(s, 100) / 100) * svgHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  line.setAttribute('points', points);
}

// ── Loading / Error States ──────────────────────────────────

export function showLoading(message) {
  const overlay = document.getElementById('loading-overlay');
  const msgEl = document.getElementById('loading-message');
  if (overlay) overlay.hidden = false;
  if (msgEl && message) msgEl.textContent = message;
}

export function hideLoading() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.hidden = true;
}

export function showError(message) {
  const errorBox = document.getElementById('video-error');
  const errorMsg = document.getElementById('error-message');
  const placeholder = document.getElementById('video-placeholder');

  if (errorBox) errorBox.hidden = false;
  if (errorMsg) errorMsg.textContent = message;
  if (placeholder) placeholder.hidden = true;
}
