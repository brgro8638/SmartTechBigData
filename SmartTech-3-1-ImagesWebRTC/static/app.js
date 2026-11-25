(() => {
  const $ = (id) => document.getElementById(id);
  const img = $('img'), viewer = $('viewer');

  let dc = null;
  let info = null;                       
  let wc = 40, ww = 400;                
  let meta = null, chunks = [], received = 0;   
  let sentAt = 0, playing = false, sliderActive = false;

  // WebRTC connection
  function setStatus(text, cls) {
    const el = $('status');
    el.textContent = text;
    el.className = 'status ' + (cls || '');
  }

  function iceComplete(pc) {
    return new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') return resolve();
      const check = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', check);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', check);
      setTimeout(resolve, 3000); // don't wait forever
    });
  }

  async function connect() {
    const pc = new RTCPeerConnection({ iceServers: [] });
    // Ordered + reliable (the default): chunks arrive in order and nothing is lost,
    // so the image the phone shows is exactly what the backend rendered.
    dc = pc.createDataChannel('images');
    dc.binaryType = 'arraybuffer';
    dc.onopen = () => { setStatus('connected (WebRTC data channel)', 'ok'); send({ type: 'hello' }); };
    dc.onclose = () => setStatus('disconnected: reload the page', 'bad');
    dc.onmessage = onMessage;
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed') setStatus('connection failed (see README)', 'bad');
    };

    await pc.setLocalDescription(await pc.createOffer());
    await iceComplete(pc);
    const res = await fetch('/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    await pc.setRemoteDescription(await res.json());
  }

  function send(obj) {
    if (!dc || dc.readyState !== 'open') return;
    sentAt = performance.now();
    dc.send(JSON.stringify(obj));
  }

  // Receiving images
  function onMessage(ev) {
    if (typeof ev.data === 'string') {
      const m = JSON.parse(ev.data);
      if (m.type === 'info') return onInfo(m);
      if (m.type === 'image') { meta = m; chunks = []; received = 0; }
      return;
    }
    if (!meta) return;
    chunks.push(ev.data);
    received += ev.data.byteLength;
    if (received >= meta.size) {
      show(new Blob(chunks, { type: meta.mime }), meta);
      meta = null; chunks = [];
    }
  }

  function show(blob, m) {
    const old = img.src;
    img.src = URL.createObjectURL(blob);
    if (old.startsWith('blob:')) URL.revokeObjectURL(old); // don't leak memory

    if (!sliderActive) $('slice').value = m.index;
    $('sliceLabel').textContent = `${m.index + 1} / ${info ? info.count : '?'}`;
    $('hud').textContent = `${m.w}×${m.h}\nslice ${m.index + 1}\nWC ${Math.round(wc)}  WW ${Math.round(ww)}`;
    $('area').textContent = m.area ? `Segmented area: ${m.area} px` : '';
    const rtt = playing ? '' : ` · round trip ${Math.round(performance.now() - sentAt)} ms`;
    $('stats').textContent = `${(blob.size / 1024).toFixed(0)} KB · render ${m.ms} ms${rtt}`;
  }

  function onInfo(m) {
    info = m;
    wc = m.wc; ww = m.ww;
    $('slice').max = m.count - 1;
    $('slice').value = m.index;
    const sel = $('filter');
    sel.innerHTML = '';
    for (const f of m.filters) {
      const o = document.createElement('option');
      o.value = f;
      o.textContent = f[0].toUpperCase() + f.slice(1);
      sel.appendChild(o);
    }
    $('wlLabel').textContent = `(${Math.round(wc)} / ${Math.round(ww)})`;
    setStatus(`connected · ${m.source}`, 'ok');
  }

  // Controls
  const set = (obj) => send({ type: 'set', ...obj });

  const slice = $('slice');
  slice.addEventListener('pointerdown', () => { sliderActive = true; });
  slice.addEventListener('pointerup', () => { sliderActive = false; });
  slice.addEventListener('input', () => set({ index: +slice.value }));
  $('prev').onclick = () => set({ index: Math.max(0, +slice.value - 1) });
  $('next').onclick = () => set({ index: Math.min(+slice.max, +slice.value + 1) });

  $('play').onclick = () => {
    playing = !playing;
    $('play').textContent = playing ? '❚❚ Pause' : '▶ Play';
    send(playing ? { type: 'play', fps: 8 } : { type: 'pause' });
  };

  function setWL(newWc, newWw) {
    wc = newWc; ww = Math.max(1, newWw);
    $('wlLabel').textContent = `(${Math.round(wc)} / ${Math.round(ww)})`;
    set({ wc, ww });
  }
  document.querySelectorAll('button[data-wc]').forEach((b) => {
    b.onclick = () => setWL(+b.dataset.wc, +b.dataset.ww);
  });
  $('wlDefault').onclick = () => info && setWL(info.wc, info.ww);

  $('filter').onchange = (e) => set({ filter: e.target.value });
  $('seg').onchange = (e) => set({ seg: e.target.value });
  $('clearSeed').onclick = () => set({ seed: null });
  $('tol').oninput = (e) => { $('tolLabel').textContent = e.target.value; set({ tol: +e.target.value }); };
  $('fmt').onchange = (e) => set({ fmt: e.target.value });

  // Touch 
  const tool = () => document.querySelector('input[name=tool]:checked').value;
  let drag = null, wlPending = false;

  viewer.addEventListener('pointerdown', (e) => {
    viewer.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, wc, ww, moved: false };
  });

  viewer.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 6) drag.moved = true;
    if (tool() === 'wl' && drag.moved && !wlPending) {
      wlPending = true;
      requestAnimationFrame(() => {      // at most one update per frame
        wlPending = false;
        if (drag) setWL(drag.wc + dy * (drag.ww / 200), drag.ww + dx * (drag.ww / 200));
      });
    }
  });

  viewer.addEventListener('pointerup', (e) => {
    const d = drag; drag = null;
    if (!d || d.moved || tool() !== 'seed') return;
    // Convert the tap from screen pixels to pixel coordinates of the actual image.
    const r = img.getBoundingClientRect();
    const x = Math.round((e.clientX - r.left) * img.naturalWidth / r.width);
    const y = Math.round((e.clientY - r.top) * img.naturalHeight / r.height);
    if (x < 0 || y < 0 || x >= img.naturalWidth || y >= img.naturalHeight) return;
    const seg = $('seg');
    if (seg.value === 'none') seg.value = 'region';
    set({ seed: [x, y], seg: seg.value });
  });
  viewer.addEventListener('pointercancel', () => { drag = null; });

  connect().catch((err) => { console.error(err); setStatus('could not connect: ' + err.message, 'bad'); });
})();
