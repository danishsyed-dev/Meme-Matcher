import memeData from './meme-data.json';

// ── Weights — must match Python config.py defaults ───────────────
const FEATURE_WEIGHTS = {
  surprise_score: 20,
  mouth_openness: 20,
  hand_raised: 20,
  eye_openness: 15,
  smile_score: 15,
};

// ── Scale factors — normalise raw diffs before decay (from Python matcher.py) ──
const FEATURE_SCALES = {
  eye_openness: 0.15,
  mouth_openness: 0.4,
  eyebrow_height: 0.05,
  head_tilt: 15.0,
  hand_raised: 1.0,
  num_hands: 1.0,
  surprise_score: 0.015,
  smile_score: 0.5,
};

const DECAY_FACTOR = 5.0;
const DEBOUNCE_FRAMES = 5;

let memes = [];
let establishedMatch = null;
let candidateMatch = null;
let candidateFrames = 0;

export async function initMatcher() {
  memes = memeData;
  return memes.length;
}

export function findMatch(userFeatures) {
  if (!userFeatures || memes.length === 0) {
    return { meme: null, score: 0 };
  }

  let bestMeme = null;
  let bestScore = -1;

  for (const meme of memes) {
    let score = 0;

    for (const [feature, weight] of Object.entries(FEATURE_WEIGHTS)) {
      const userFeat = userFeatures[feature] ?? 0;
      const memeFeat = meme.features[feature] ?? 0;
      const diff = Math.abs(userFeat - memeFeat);
      const scale = FEATURE_SCALES[feature] ?? 1.0;
      const normDiff = diff / scale;
      score += weight * Math.exp(-normDiff * DECAY_FACTOR);
    }

    if (score > bestScore) {
      bestScore = score;
      bestMeme = meme;
    }
  }

  const maxPossibleScore = Object.values(FEATURE_WEIGHTS).reduce((a, b) => a + b, 0);
  const normalizedScore = (bestScore / maxPossibleScore) * 100;

  if (candidateMatch && candidateMatch.filename === bestMeme.filename) {
    candidateFrames++;
  } else {
    candidateMatch = bestMeme;
    candidateFrames = 1;
  }

  if (candidateFrames >= DEBOUNCE_FRAMES) {
    establishedMatch = candidateMatch;
  }

  return { meme: establishedMatch, score: establishedMatch ? normalizedScore : 0 };
}

export function getMemes() {
  return memes;
}
