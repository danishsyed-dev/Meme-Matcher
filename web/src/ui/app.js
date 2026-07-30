/**
 * Main application controller.
 * Orchestrates camera, detection, matching, and UI updates.
 */

import { initDetector, detectFrame } from '../detection/detector.js';
import { extractFeatures } from '../detection/features.js';
import { initMatcher, findMatch, getMemes } from '../matching/matcher.js';
import { startCamera, isCameraActive } from './camera.js';
import {
  updateMatch, clearMatch, updateFps, updateCameraState, updateMemeCount,
  updateGallery, updateHistory, showLoading, hideLoading, showError
} from './panels.js';
import { initControls, captureScreenshot } from './controls.js';

/**
 * Initializes the entire application:
 * 1. Loads MediaPipe models
 * 2. Loads meme feature data
 * 3. Starts the webcam
 * 4. Begins the real-time detection loop
 */
export async function initApp() {
  try {
    showLoading('Initializing MediaPipe models…');
    await initDetector();

    showLoading('Loading meme features…');
    const count = await initMatcher();
    updateMemeCount(count);

    const videoElement = document.getElementById('webcam');
    if (!videoElement) {
      throw new Error('Webcam video element not found in the page.');
    }

    showLoading('Starting camera…');
    await startCamera(videoElement);
    updateCameraState(isCameraActive());

    // Render the meme gallery strip
    updateGallery(getMemes(), null);

    hideLoading();

    // Wire up button controls and keyboard shortcuts
    initControls({
      onScreenshot: () => captureScreenshot(videoElement),
      onReload: () => window.location.reload(),
    });

    // ── Detection Loop ─────────────────────────────────────
    let lastFpsUpdate = performance.now();
    let frameCount = 0;

    function loop() {
      try {
        const timestamp = performance.now();
        const result = detectFrame(videoElement, timestamp);

        if (result && result.faceLandmarks && result.faceLandmarks.length > 0) {
          const faceLandmarks = result.faceLandmarks[0];
          const handLandmarks = result.handLandmarks || [];

          const features = extractFeatures(faceLandmarks, handLandmarks);
          if (features) {
            const match = findMatch(features);
            if (match.meme) {
              updateMatch(match.meme, match.score);
              updateGallery(getMemes(), match.meme);
            } else {
              clearMatch();
            }
            updateHistory(match.score);
          }
        }

        // FPS calculation — throttled to every 500ms
        frameCount++;
        if (timestamp - lastFpsUpdate >= 500) {
          const fps = frameCount / ((timestamp - lastFpsUpdate) / 1000);
          updateFps(fps);
          frameCount = 0;
          lastFpsUpdate = timestamp;
        }
      } catch (err) {
        console.error('Detection loop error:', err);
      }

      requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);

  } catch (err) {
    console.error('Failed to initialize app:', err);
    showError(err.message);
    hideLoading();
  }
}
