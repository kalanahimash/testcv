const preview = document.querySelector('#preview');
const context = preview.getContext('2d');
const placeholder = document.querySelector('#placeholder');
const start = document.querySelector('#start');
const stop = document.querySelector('#stop');
const camera = document.querySelector('#camera');
const detect = document.querySelector('#detect');
const confidence = document.querySelector('#confidence');
const status = document.querySelector('#status');
const indicator = document.querySelector('#indicator');
const detectionResult = document.querySelector('#detection-result');
const demo = document.querySelector('#demo');
const signs = document.querySelector('#signs');
const signDemo = document.querySelector('#sign-demo');
const signResult = document.querySelector('#sign-result');
let controller = null;

function showDetections(enabled, result) {
  detectionResult.textContent = enabled
    ? `Detection running · ${result.count} object(s) · ${result.fps} inference FPS. ${result.labels.length ? result.labels.join(', ') : 'No objects found: try better front lighting or a lower confidence threshold.'}`
    : 'Detection is OFF. Stop the camera and check Detect road objects to enable it.';
  const signLabels = (result.signs || []).map(item => `${item.name} ${Math.round(item.confidence * 100)}%`);
  signResult.textContent = result.signs_enabled
    ? `Sign detector running · ${signLabels.length ? signLabels.join(', ') : 'No recognized signs in this frame.'}`
    : 'Sign detection is OFF.';
}

function message(text, error = false) {
  status.textContent = text;
  status.classList.toggle('error', error);
}
function reset() {
  preview.hidden = true;
  placeholder.hidden = false;
  start.disabled = false;
  stop.disabled = true;
  camera.disabled = false;
  detect.disabled = false;
  confidence.disabled = false;
  demo.disabled = false;
  signs.disabled = false;
  signDemo.disabled = false;
  indicator.textContent = '● OFFLINE';
  indicator.classList.remove('live');
}
stop.addEventListener('click', () => {
  controller?.abort();
  message('Stopping camera…');
});
document.querySelector('#mirror').addEventListener('change', event => {
  preview.classList.toggle('mirrored', event.target.checked);
});
window.addEventListener('pagehide', () => controller?.abort());

start.addEventListener('click', async () => {
  const session = new AbortController();
  controller = session;
  let drawBitmap = null;   // holds latest decoded ImageBitmap waiting for next vsync
  let rafPending = false;  // true while a requestAnimationFrame is already queued
  start.disabled = true;
  camera.disabled = true;
  detect.disabled = true;
  confidence.disabled = true;
  demo.disabled = true;
  signs.disabled = true;
  signDemo.disabled = true;
  stop.disabled = false;
  message(detect.checked || signs.checked ? 'Opening camera and loading detection models…' : 'Opening camera…');
  detectionResult.textContent = 'Waiting for the first processed frame…';
  try {
    const response = await fetch(`/stream?camera=${camera.value}&detect=${detect.checked ? 1 : 0}&signs=${signs.checked ? 1 : 0}&confidence=${confidence.value}`, {signal: session.signal});
    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.error || 'Could not start camera.');
    }
    const reader = response.body.getReader();
    let buffer = new Uint8Array(0);
    const decoder = new TextDecoder();
    while (true) {
      const {value, done} = await reader.read();
      if (done) throw new Error('Stream stopped. Check the camera and Python terminal, then start again.');
      const joined = new Uint8Array(buffer.length + value.length);
      joined.set(buffer);
      joined.set(value, buffer.length);
      buffer = joined;
      // Each MJPEG part has headers and an exact JPEG byte count.
      while (buffer.length) {
        let headerEnd = -1;
        for (let i = 0; i < buffer.length - 3; i++) {
          if (buffer[i] === 13 && buffer[i+1] === 10 && buffer[i+2] === 13 && buffer[i+3] === 10) {headerEnd = i; break;}
        }
        if (headerEnd < 0) break;
        const header = decoder.decode(buffer.subarray(0, headerEnd));
        const detectionMode = /X-Detection:\s*(on|off)/i.exec(header);
        const detectionData = /X-Detection-Result:\s*([^\r\n]+)/i.exec(header);
        if (!detectionMode || !detectionData) throw new Error('An older server is streaming. Stop all copies of app.py, restart it, then refresh this page.');
        if ((detect.checked || signs.checked) && detectionMode[1] !== 'on') throw new Error('The server did not enable detection. Restart app.py and retry.');
        if (signs.checked && !JSON.parse(detectionData[1]).signs_enabled) throw new Error('This server has not enabled sign detection. Restart app.py and refresh.');
        const match = /Content-Length:\s*(\d+)/i.exec(header);
        if (!match) throw new Error('Invalid camera stream.');
        const length = Number(match[1]);
        const end = headerEnd + 4 + length;
        if (buffer.length < end) break;
        const frame = await createImageBitmap(new Blob([buffer.slice(headerEnd + 4, end)], {type:'image/jpeg'}));
        if (session.signal.aborted) {frame.close(); message('Camera is off.'); return;}
        if (preview.width !== frame.width || preview.height !== frame.height) {
          preview.width = frame.width;
          preview.height = frame.height;
        }
        // Swap in the newest decoded frame; drop the previous one if it wasn't drawn yet
        if (drawBitmap) { drawBitmap.close(); drawBitmap = null; }
        drawBitmap = frame;
        // Schedule exactly one draw per vsync cycle – always paints the freshest bitmap
        if (!rafPending) {
          rafPending = true;
          requestAnimationFrame(() => {
            rafPending = false;
            if (drawBitmap && !session.signal.aborted) {
              context.drawImage(drawBitmap, 0, 0);
              drawBitmap.close();
              drawBitmap = null;
            }
          });
        }
        showDetections(detectionMode[1] === 'on', JSON.parse(detectionData[1]));
        buffer = buffer.slice(end);
        preview.hidden = false;
        placeholder.hidden = true;
        indicator.textContent = '● LIVE';
        indicator.classList.add('live');
        message(`Streaming camera ${camera.value} · ${preview.width} × ${preview.height}${detect.checked ? ' · Road detection ON' : ''}`);
      }
    }
  } catch (error) {
    if (session.signal.aborted) message('Camera is off.');
    else message(error.message, true);
    detectionResult.textContent = session.signal.aborted ? 'Detection stopped.' : 'Detection interrupted. See the message below.';
    signResult.textContent = 'Sign detection stopped.';
  } finally {
    if (drawBitmap) { drawBitmap.close(); drawBitmap = null; }  // release any undisplayed frame
    session.abort();
    controller = null;
    reset();
  }
});

async function runDemo(testSigns = false) {
  reset();
  demo.disabled = true;
  signDemo.disabled = true;
  start.disabled = true;
  message(testSigns ? 'Testing crossing-sign recognition and speed-number reading…' : 'Running detection on the bundled road sample…');
  try {
    const response = await fetch(`/detection-demo?signs=${testSigns ? 1 : 0}`);
    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.error || 'Sample test failed.');
    }
    const result = JSON.parse(response.headers.get('X-Detection-Result'));
    const frame = await createImageBitmap(await response.blob());
    preview.width = frame.width;
    preview.height = frame.height;
    context.drawImage(frame, 0, 0);
    frame.close();
    preview.hidden = false;
    placeholder.hidden = true;
    indicator.textContent = 'SAMPLE TEST';
    showDetections(true, result);
    message(`Sample test complete: ${result.count} detections. ${testSigns ? 'These are synthetic test signs, not road validation images.' : 'This is a test photo, not the live camera.'}`);
  } catch (error) {
    message(error.message, true);
    detectionResult.textContent = 'Sample test failed.';
  } finally {
    demo.disabled = false;
    signDemo.disabled = false;
    start.disabled = false;
  }
}
demo.addEventListener('click', () => runDemo(false));
signDemo.addEventListener('click', () => runDemo(true));
