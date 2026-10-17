/*
 * JARVIS desktop — UI wiring.
 *
 * Everything your backend needs is on window.jarvis:
 *
 *   jarvis.setState('idle' | 'listening' | 'thinking' | 'speaking', caption?)
 *   jarvis.setLevel(0..1)                 // live audio volume for the face
 *   jarvis.addMessage('user' | 'jarvis', text)
 *   jarvis.addChips(['Thought · 1.2s', {text:'Spoke · Piper', voice:true}])
 *   jarvis.setStatus({ agent, agentSub, mic, voice, link })
 *   jarvis.addLatency(ms)                 // feeds the response-time sparkline
 *   jarvis.startMic()  -> Promise         // real microphone, drives the face
 *   jarvis.stopMic()
 *   jarvis.playSpeech(urlOrBlob) -> Promise  // plays TTS audio (e.g. Piper wav) and animates the face
 *   jarvis.onSend = (text) => {}          // called when the user sends a message
 *   jarvis.onCommand = (name) => {}       // palette / rail commands: memory, agents, tasks, terminal, voice, settings …
 *
 * DEMO = true fakes replies so you can try the UI with no backend.
 * Set it to false once onSend is connected to the real Jarvis.
 */
const DEMO = true;

(function () {
  const $ = (s) => document.querySelector(s);
  const face = $('#face');
  const label = $('#state-label');
  const caption = $('#caption');
  const messages = $('#messages');

  const DEFAULT_CAPTIONS = {
    idle: 'Say “Jarvis” or hold Space to talk.',
    listening: 'Listening…',
    thinking: 'Working on it…',
    speaking: '',
  };

  // ---------- face state ----------
  face.addEventListener('statechange', (e) => {
    label.textContent = e.detail.label;
    document.documentElement.style.setProperty('--accent', e.detail.color);
  });

  function setState(state, text) {
    face.state = state;
    caption.textContent = text ?? DEFAULT_CAPTIONS[state] ?? '';
    setStatus({
      agent: state === 'thinking' ? 'Working' : 'Idle',
      voice: state === 'speaking' ? 'Speaking' : 'Idle',
      mic: state === 'listening' ? 'Live' : (micStream ? 'Paused' : 'Off'),
    });
  }

  function setStatus(s) {
    if (s.agent != null) $('#st-agent').textContent = s.agent;
    if (s.agentSub != null) $('#st-agent-sub').textContent = s.agentSub;
    if (s.mic != null) $('#st-mic').textContent = s.mic;
    if (s.voice != null) $('#st-voice').textContent = s.voice;
    if (s.link != null) $('#link-text').textContent = s.link;
  }

  // ---------- messages ----------
  function addMessage(role, text) {
    const el = document.createElement('div');
    if (role === 'user') {
      el.className = 'msg-user';
      el.textContent = text;
    } else {
      el.className = 'msg-jarvis';
      const span = document.createElement('span');
      span.textContent = text;
      el.appendChild(span);
    }
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  function addChips(list) {
    const row = document.createElement('div');
    row.className = 'chips';
    list.forEach((c) => {
      const chip = document.createElement('span');
      const item = typeof c === 'string' ? { text: c } : c;
      chip.className = 'chip' + (item.voice ? ' voice' : '');
      chip.textContent = item.text;
      row.appendChild(chip);
    });
    messages.appendChild(row);
    messages.scrollTop = messages.scrollHeight;
  }

  // ---------- latency sparkline ----------
  const latencies = [];
  function addLatency(ms) {
    latencies.push(ms);
    if (latencies.length > 20) latencies.shift();
    const max = Math.max(...latencies, 1);
    const step = latencies.length > 1 ? 120 / (latencies.length - 1) : 0;
    $('#spark').setAttribute('points', latencies.map((v, i) => `${(i * step).toFixed(1)},${(26 - (v / max) * 22).toFixed(1)}`).join(' '));
    $('#st-latency').textContent = `${Math.round(ms)} ms last reply`;
  }

  // ---------- audio level helpers ----------
  let audioCtx = null;
  function ctx() { return (audioCtx ||= new (window.AudioContext || window.webkitAudioContext)()); }

  function meter(sourceNode, whileFn) {
    const analyser = ctx().createAnalyser();
    analyser.fftSize = 512;
    sourceNode.connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);
    (function loop() {
      if (!whileFn()) return;
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      face.setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 4)); // RMS, boosted
      requestAnimationFrame(loop);
    })();
    return analyser;
  }

  // real microphone
  let micStream = null;
  async function startMic() {
    if (micStream) return;
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const src = ctx().createMediaStreamSource(micStream);
    setState('listening');
    meter(src, () => !!micStream && face.state === 'listening');
  }
  function stopMic() {
    if (!micStream) return;
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
    setStatus({ mic: 'Off' });
  }

  // TTS playback (Piper output: a URL, Blob or ArrayBuffer of a wav file)
  function playSpeech(audio, text) {
    return new Promise((resolve) => {
      const url = audio instanceof Blob ? URL.createObjectURL(audio)
        : audio instanceof ArrayBuffer ? URL.createObjectURL(new Blob([audio], { type: 'audio/wav' }))
        : audio;
      const el = new Audio(url);
      const src = ctx().createMediaElementSource(el);
      src.connect(ctx().destination);
      let playing = true;
      meter(src, () => playing);
      setState('speaking', text);
      el.onended = el.onerror = () => { playing = false; setState('idle'); resolve(); };
      ctx().resume().then(() => el.play());
    });
  }

  // ---------- composer ----------
  const input = $('#input');
  function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage('user', text);
    if (typeof jarvis.onSend === 'function') jarvis.onSend(text);
    else if (DEMO) demoReply(text);
  }
  $('#composer').addEventListener('submit', (e) => { e.preventDefault(); send(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  // ---------- push to talk (mic button or hold Space) ----------
  const micBtn = $('#mic-btn');
  async function pttDown() {
    micBtn.setAttribute('aria-pressed', 'true');
    try { await startMic(); } catch (err) { setState('listening', 'Microphone blocked: using demo animation'); }
  }
  function pttUp() {
    micBtn.setAttribute('aria-pressed', 'false');
    stopMic();
    if (typeof jarvis.onPushToTalkEnd === 'function') jarvis.onPushToTalkEnd();
    else if (DEMO) demoReply('(voice message)');
    else setState('idle');
  }
  micBtn.addEventListener('pointerdown', pttDown);
  micBtn.addEventListener('pointerup', pttUp);
  micBtn.addEventListener('pointerleave', () => { if (micBtn.getAttribute('aria-pressed') === 'true') pttUp(); });

  const typing = () => ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  let spaceHeld = false;
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !typing() && !palette.open && !spaceHeld) { e.preventDefault(); spaceHeld = true; pttDown(); }
  });
  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && spaceHeld) { spaceHeld = false; pttUp(); }
  });

  // ---------- command palette ----------
  const palette = $('#palette');
  const pInput = $('#palette-input');
  const cmds = () => [...palette.querySelectorAll('.cmd')].filter((c) => !c.hidden);
  let sel = 0;
  function highlight() { cmds().forEach((c, i) => c.classList.toggle('sel', i === sel)); }
  function openPalette() { pInput.value = ''; filter(); palette.showModal(); pInput.focus(); }
  function filter() {
    const q = pInput.value.toLowerCase();
    palette.querySelectorAll('.cmd').forEach((c) => { c.hidden = !c.textContent.toLowerCase().includes(q); });
    sel = 0; highlight();
  }
  pInput.addEventListener('input', filter);
  pInput.addEventListener('keydown', (e) => {
    const list = cmds();
    if (e.key === 'ArrowDown') { sel = Math.min(sel + 1, list.length - 1); highlight(); e.preventDefault(); }
    if (e.key === 'ArrowUp') { sel = Math.max(sel - 1, 0); highlight(); e.preventDefault(); }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (list[sel]) list[sel].click();
      else if (pInput.value.trim()) { palette.close(); input.value = pInput.value; send(); } // free text = ask Jarvis
    }
  });
  $('#open-palette').addEventListener('click', openPalette);
  palette.addEventListener('click', (e) => { if (e.target === palette) palette.close(); });

  // ---------- agents drawer ----------
  const drawer = $('#drawer');
  function toggleDrawer(open) {
    drawer.classList.toggle('open', open);
    drawer.setAttribute('aria-hidden', String(!open));
  }
  $('#close-drawer').addEventListener('click', () => toggleDrawer(false));

  // ---------- commands (palette + rail) ----------
  function runCommand(name) {
    if (palette.open) palette.close();
    if (name === 'agents') toggleDrawer(!drawer.classList.contains('open'));
    if (typeof jarvis.onCommand === 'function') jarvis.onCommand(name);
    else if (DEMO && name !== 'agents') caption.textContent = `“${name}” isn’t connected yet.`;
  }
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-cmd]');
    if (el) { e.preventDefault(); runCommand(el.dataset.cmd); }
  });

  // ---------- global shortcuts ----------
  document.addEventListener('keydown', (e) => {
    if (!(e.ctrlKey || e.metaKey)) { if (e.key === 'Escape') toggleDrawer(false); return; }
    const k = e.key.toLowerCase();
    const map = { k: null, m: 'memory', g: 'agents', t: 'tasks', '`': 'terminal', ',': 'settings' };
    if (!(k in map)) return;
    e.preventDefault();
    if (k === 'k') palette.open ? palette.close() : openPalette();
    else runCommand(map[k]);
  });

  // ---------- clock ----------
  function tick() {
    const d = new Date();
    $('#clock').textContent = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' · ' +
      d.toLocaleDateString([], { weekday: 'short', day: '2-digit', month: 'short' }).toUpperCase();
  }
  tick(); setInterval(tick, 15000);

  // ---------- demo replies (no backend) ----------
  function demoReply(text) {
    const t0 = performance.now();
    setState('thinking', 'Working on it…');
    setTimeout(() => {
      addChips(['Thought · 1.2s', { text: 'Spoke · Piper', voice: true }]);
      const reply = 'Demo mode: connect jarvis.onSend to your backend and I’ll answer for real.';
      addMessage('jarvis', reply);
      addLatency(performance.now() - t0);
      setState('speaking', reply);
      // fake speech levels so the bars move
      const start = performance.now();
      (function talk() {
        const t = (performance.now() - start) / 1000;
        if (t > 3 || face.state !== 'speaking') { if (face.state === 'speaking') setState('idle'); return; }
        face.setLevel(0.35 + 0.35 * Math.abs(Math.sin(t * 9)) * Math.abs(Math.sin(t * 2.3)) + Math.random() * 0.2);
        requestAnimationFrame(talk);
      })();
    }, 1200);
  }

  window.jarvis = {
    setState, setLevel: (v) => face.setLevel(v), setStatus,
    addMessage, addChips, addLatency,
    startMic, stopMic, playSpeech,
    openPalette, toggleDrawer,
    onSend: null, onCommand: null, onPushToTalkEnd: null,
  };
})();
