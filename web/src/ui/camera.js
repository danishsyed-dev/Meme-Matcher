let stream = null;

export async function startCamera(videoElement) {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
    videoElement.srcObject = stream;
    return new Promise((resolve) => {
      videoElement.onloadeddata = () => {
        resolve(true);
      };
    });
  } catch (error) {
    console.error('Camera access error:', error);
    if (error.name === 'NotAllowedError') {
      throw new Error('Camera access was denied by the user.');
    } else if (error.name === 'NotFoundError') {
      throw new Error('No camera device was found.');
    } else {
      throw new Error(`Failed to access camera: ${error.message}`);
    }
  }
}

export function stopCamera(videoElement) {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
    stream = null;
  }
}

export function isCameraActive() {
  return stream !== null && stream.active;
}
