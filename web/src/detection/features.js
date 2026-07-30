/**
 * Computes features from face and hand landmarks for Meme Matcher.
 */

const LEFT_EYE_UPPER = [159, 145, 158];
const LEFT_EYE_LOWER = [23, 27, 133];
const RIGHT_EYE_UPPER = [386, 374, 385];
const RIGHT_EYE_LOWER = [253, 257, 362];
const LEFT_EYEBROW = [70, 63, 105, 66, 107];
const RIGHT_EYEBROW = [300, 293, 334, 296, 336];
const MOUTH_TOP = 13;
const MOUTH_BOTTOM = 14;
const MOUTH_LEFT = 61;
const MOUTH_RIGHT = 291;

/**
 * Helper to get a point {x, y} from the landmarks array.
 * @param {Array} landmarks Array of landmark objects.
 * @param {number} idx The index of the landmark.
 * @returns {Object} {x, y} point.
 */
function pt(landmarks, idx) {
    return {
        x: landmarks[idx].x,
        y: landmarks[idx].y
    };
}

/**
 * Calculates Euclidean distance between two points.
 * @param {Object} a {x, y} point.
 * @param {Object} b {x, y} point.
 * @returns {number} Distance.
 */
function dist(a, b) {
    return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
}

/**
 * Calculates the Eye Aspect Ratio (EAR) for given landmarks.
 * @param {Array} landmarks Array of landmarks.
 * @param {Array} upperIds Indices for upper eye landmarks.
 * @param {Array} lowerIds Indices for lower eye landmarks.
 * @returns {number} The EAR.
 */
function eyeAspectRatio(landmarks, upperIds, lowerIds) {
    let verticalSum = 0;
    for (let i = 0; i < upperIds.length; i++) {
        verticalSum += dist(pt(landmarks, upperIds[i]), pt(landmarks, lowerIds[i]));
    }
    const verticalDist = verticalSum / upperIds.length;
    
    // Approximate horizontal width as the distance between the first and last points in lowerIds
    const horizontalDist = dist(pt(landmarks, lowerIds[0]), pt(landmarks, lowerIds[lowerIds.length - 1])) + 1e-6;
    
    return verticalDist / horizontalDist;
}

/**
 * Calculates the center position of a set of landmarks.
 * @param {Array} landmarks Array of landmarks.
 * @param {Array} indices Indices to average.
 * @returns {Object} {x, y} center point.
 */
function getCenter(landmarks, indices) {
    let sumX = 0;
    let sumY = 0;
    for (const idx of indices) {
        const p = pt(landmarks, idx);
        sumX += p.x;
        sumY += p.y;
    }
    return { 
        x: sumX / indices.length, 
        y: sumY / indices.length 
    };
}

/**
 * Extracts normalized features from face and hand landmarks.
 * @param {Array} faceLandmarks Array of face landmarks (results.faceLandmarks).
 * @param {Array} handLandmarks Array of hand landmarks (results.landmarks).
 * @returns {Object|null} Extracted feature object or null if no face is detected.
 */
export function extractFeatures(faceLandmarks, handLandmarks) {
    if (!faceLandmarks || faceLandmarks.length === 0) {
        return null;
    }

    // faceLandmarks is already a single face's landmark array
    // (app.js passes result.faceLandmarks[0])
    const face = faceLandmarks;
    
    // 1. eye_openness
    const leftEar = eyeAspectRatio(face, LEFT_EYE_UPPER, LEFT_EYE_LOWER);
    const rightEar = eyeAspectRatio(face, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER);
    const eye_openness = (leftEar + rightEar) / 2.0;

    // 2. mouth_openness
    const mouthHeight = dist(pt(face, MOUTH_TOP), pt(face, MOUTH_BOTTOM));
    const mouthWidth = dist(pt(face, MOUTH_LEFT), pt(face, MOUTH_RIGHT)) + 1e-6;
    const mouth_openness = mouthHeight / mouthWidth;

    // 3. eyebrow_height
    const leftEyeCenter = getCenter(face, [...LEFT_EYE_UPPER, ...LEFT_EYE_LOWER]);
    const rightEyeCenter = getCenter(face, [...RIGHT_EYE_UPPER, ...RIGHT_EYE_LOWER]);
    const leftEyebrowCenter = getCenter(face, LEFT_EYEBROW);
    const rightEyebrowCenter = getCenter(face, RIGHT_EYEBROW);
    
    const leftEyebrowHeight = dist(leftEyebrowCenter, leftEyeCenter);
    const rightEyebrowHeight = dist(rightEyebrowCenter, rightEyeCenter);
    const eyebrow_height = (leftEyebrowHeight + rightEyebrowHeight) / 2.0;

    // 4. head_tilt
    const dx = rightEyeCenter.x - leftEyeCenter.x;
    const dy = rightEyeCenter.y - leftEyeCenter.y;
    const head_tilt = Math.atan2(dy, dx) * (180.0 / Math.PI);

    // 5. num_hands and hand_raised
    let hand_raised = 0.0;
    const num_hands = handLandmarks ? handLandmarks.length : 0;
    
    if (num_hands > 0) {
        // Find top of the face (minimum y coordinate)
        let faceTopY = face[0].y;
        for (let i = 1; i < face.length; i++) {
            if (face[i].y < faceTopY) {
                faceTopY = face[i].y;
            }
        }

        // Check if any wrist (landmark 0) is above face top + 0.3
        for (const hand of handLandmarks) {
            if (hand[0].y < faceTopY + 0.3) {
                hand_raised = 1.0;
                break;
            }
        }
    }

    // 6. surprise_score
    const surprise_score = eye_openness * eyebrow_height * mouth_openness;

    // 7. smile_score
    const smile_score = Math.max(0, Math.min(1, 1 - mouth_openness));

    return {
        eye_openness,
        mouth_openness,
        eyebrow_height,
        head_tilt,
        hand_raised,
        num_hands,
        surprise_score,
        smile_score
    };
}
