import { FilesetResolver, FaceLandmarker, HandLandmarker } from '@mediapipe/tasks-vision';

let faceLandmarker = null;
let handLandmarker = null;

/**
 * Initializes the MediaPipe Face and Hand landmarkers.
 * @returns {Promise<boolean>} True if initialization was successful.
 */
export async function initDetector() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: '/models/face_landmarker.task',
            },
            numFaces: 1,
            runningMode: 'VIDEO',
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false
        });

        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: '/models/hand_landmarker.task',
            },
            numHands: 2,
            runningMode: 'VIDEO',
            minHandDetectionConfidence: 0.3,
            minHandPresenceConfidence: 0.3,
            minTrackingConfidence: 0.5
        });

        return true;
    } catch (error) {
        console.error('Error initializing detectors:', error);
        return false;
    }
}

/**
 * Runs detection on a video frame.
 * @param {HTMLVideoElement} videoElement The video element to process.
 * @param {number} timestampMs The timestamp of the current frame in milliseconds.
 * @returns {Object|null} Object containing faceLandmarks and handLandmarks, or null on error.
 */
export function detectFrame(videoElement, timestampMs) {
    if (!faceLandmarker || !handLandmarker) {
        return null;
    }

    try {
        const faceResult = faceLandmarker.detectForVideo(videoElement, timestampMs);
        const handResult = handLandmarker.detectForVideo(videoElement, timestampMs);

        return {
            faceLandmarks: faceResult.faceLandmarks,
            handLandmarks: handResult.landmarks
        };
    } catch (error) {
        console.error('Error detecting frame:', error);
        return null;
    }
}

/**
 * Cleans up the detectors.
 */
export function closeDetector() {
    if (faceLandmarker) {
        faceLandmarker.close();
        faceLandmarker = null;
    }
    if (handLandmarker) {
        handLandmarker.close();
        handLandmarker = null;
    }
}
