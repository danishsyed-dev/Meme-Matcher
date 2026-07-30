export function initControls({ onScreenshot, onReload }) {
  const btnScreenshot = document.getElementById('btn-screenshot');
  const btnReload = document.getElementById('btn-reload');

  if (btnScreenshot) {
    btnScreenshot.addEventListener('click', () => {
      onScreenshot();
    });
  }

  if (btnReload) {
    btnReload.addEventListener('click', () => {
      onReload();
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === 's') {
      e.preventDefault();
      onScreenshot();
    }
    if (e.ctrlKey && e.key.toLowerCase() === 'r') {
      e.preventDefault();
      onReload();
    }
  });
}

export function captureScreenshot(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = `meme-matcher-${timestamp}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 'image/jpeg', 0.9);
}
