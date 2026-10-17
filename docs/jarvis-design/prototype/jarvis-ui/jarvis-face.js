/*
 * <jarvis-face> — the animated JARVIS face as a drop-in web component.
 *
 * Usage:
 *   <script src="jarvis-face.js"></script>
 *   <jarvis-face state="idle"></jarvis-face>
 *
 * API:
 *   face.state = 'idle' | 'listening' | 'thinking' | 'speaking'
 *   face.setLevel(0..1)   // live mic / speech volume; drives the ticks & bars.
 *                         // Call it every animation frame while audio flows.
 *                         // If you never call it, a built-in animation plays instead.
 *   face.addEventListener('statechange', e => e.detail) // {state, label, color}
 */
(function () {
  const STATES = {
    idle:      { color: '#3FA9D6', label: 'STANDING BY', ring: 0.5 },
    listening: { color: '#4FD1FF', label: 'LISTENING',   ring: 1 },
    thinking:  { color: '#FFB547', label: 'THINKING',    ring: 0.8 },
    speaking:  { color: '#7FE3FF', label: 'SPEAKING',    ring: 1 },
  };

  let ticks = '';
  for (let i = 0; i < 60; i++) {
    const h = 6 + ((i * 37) % 11) * 2;
    const v = (0.5 + ((i * 53) % 10) / 20).toFixed(2);
    ticks += `<g transform="rotate(${i * 6} 200 200)"><rect class="tk" x="199" y="${88 - h}" width="2" height="${h}" rx="1" style="animation-delay:${((i % 10) * 0.14).toFixed(2)}s;--v:${v}"/></g>`;
  }
  let bars = '';
  for (let j = 0; j < 36; j++) {
    const v = (0.45 + ((j * 29) % 11) / 20).toFixed(2);
    bars += `<g transform="rotate(${j * 10} 200 200)"><rect class="bar" x="198" y="66" width="4" height="34" rx="2" style="animation-delay:${(((j * 7) % 9) * 0.1).toFixed(2)}s;--v:${v}"/></g>`;
  }

  const TEMPLATE = `
<style>
  :host { display:block; aspect-ratio:1/1; --accent:#4FD1FF; --ring:1; --lvl:0; }
  svg { width:100%; height:100%; overflow:visible; color:var(--accent); transition:color .5s ease; }
  .rings { opacity:var(--ring); transition:opacity .5s ease; }
  .r1 { transform-origin:200px 200px; animation:spin 60s linear infinite; }
  .r2 { transform-origin:200px 200px; animation:spinr 28s linear infinite; }
  .r3 { transform-origin:200px 200px; animation:spin 6s linear infinite; }
  .r4 { transform-origin:200px 200px; animation:spinr 3.5s linear infinite; }
  .core { transform-origin:200px 200px; animation:breathe 3s ease-in-out infinite; }
  :host([state="idle"]) .core { animation-duration:6s; }
  .tk, .bar { fill:currentColor; transform-box:fill-box; transform-origin:center bottom; }
  .tk  { animation:tick 1.4s ease-in-out infinite; }
  .bar { animation:bar .9s ease-in-out infinite; }
  :host([driven]) .tk  { animation:none; opacity:calc(.25 + var(--lvl) * .75); transform:scaleY(calc(.4 + var(--lvl) * var(--v) * 1.8)); transition:transform 60ms linear, opacity 60ms linear; }
  :host([driven]) .bar { animation:none; transform:scaleY(calc(.15 + var(--lvl) * var(--v) * 1.7)); transition:transform 60ms linear; }
  .st { display:none; }
  :host([state="idle"]) .st-idle,
  :host([state="listening"]) .st-listening,
  :host([state="thinking"]) .st-thinking,
  :host([state="speaking"]) .st-speaking { display:inline; }
  @keyframes spin  { to { transform:rotate(360deg); } }
  @keyframes spinr { to { transform:rotate(-360deg); } }
  @keyframes breathe { 0%,100% { opacity:.5; transform:scale(.95); } 50% { opacity:1; transform:scale(1.03); } }
  @keyframes bar  { 0%,100% { transform:scaleY(.25); } 50% { transform:scaleY(1); } }
  @keyframes tick { 0%,100% { opacity:.25; } 50% { opacity:1; } }
  @media (prefers-reduced-motion: reduce) { * { animation:none !important; } }
</style>
<svg viewBox="0 0 400 400" role="img" aria-hidden="true">
  <defs>
    <radialGradient id="jf-core">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.95"/>
      <stop offset="35%" stop-color="currentColor" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="currentColor" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <g class="rings">
    <g class="r1">
      <circle cx="200" cy="200" r="192" fill="none" stroke="currentColor" stroke-dasharray="1 7"/>
      <path d="M200 4v14M200 382v14M4 200h14M382 200h14" stroke="currentColor" stroke-width="2"/>
    </g>
    <g class="r2">
      <circle cx="200" cy="200" r="166" fill="none" stroke="currentColor" stroke-width="3" stroke-dasharray="70 34 12 34" stroke-linecap="round"/>
    </g>
  </g>
  <circle cx="200" cy="200" r="138" fill="none" stroke="currentColor" stroke-opacity="0.25"/>

  <g class="st st-listening">${ticks}</g>
  <g class="st st-thinking">
    <g class="r3"><circle cx="200" cy="80" r="6" fill="currentColor"/><circle cx="200" cy="320" r="3" fill="currentColor" fill-opacity="0.6"/></g>
    <g class="r4"><circle cx="200" cy="104" r="4" fill="currentColor"/><path d="M120 200a80 80 0 0 1 80-80" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></g>
  </g>
  <g class="st st-speaking">${bars}</g>

  <g class="core"><circle cx="200" cy="200" r="96" fill="url(#jf-core)" opacity="0.55"/></g>
  <circle cx="200" cy="200" r="64" fill="#06090D" fill-opacity="0.35" stroke="currentColor" stroke-width="2"/>

  <!-- expressions -->
  <path class="st st-idle" d="M176 200h48" stroke="#E8F7FF" stroke-width="3" stroke-linecap="round"/>
  <g class="st st-listening"><circle cx="184" cy="198" r="5" fill="#E8F7FF"/><circle cx="216" cy="198" r="5" fill="#E8F7FF"/></g>
  <g class="st st-thinking"><circle cx="184" cy="200" r="4" fill="#FFF3DC"/><circle cx="200" cy="200" r="4" fill="#FFF3DC" fill-opacity="0.7"/><circle cx="216" cy="200" r="4" fill="#FFF3DC" fill-opacity="0.4"/></g>
  <path class="st st-speaking" d="M174 200h6l4-10 6 20 6-26 6 30 6-22 4 8h12" fill="none" stroke="#E8F7FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

  class JarvisFace extends HTMLElement {
    static get observedAttributes() { return ['state']; }

    constructor() {
      super();
      this.attachShadow({ mode: 'open' }).innerHTML = TEMPLATE;
      this._decay = null;
    }

    connectedCallback() {
      this.setAttribute('role', 'img');
      if (!this.hasAttribute('state')) this.setAttribute('state', 'idle');
      else this._apply();
    }

    attributeChangedCallback(name, oldV, newV) {
      if (oldV !== newV) this._apply();
    }

    get state() { return this.getAttribute('state') || 'idle'; }
    set state(v) { this.setAttribute('state', STATES[v] ? v : 'idle'); }

    setLevel(v) {
      v = Math.max(0, Math.min(1, Number(v) || 0));
      this.style.setProperty('--lvl', v.toFixed(3));
      if (!this.hasAttribute('driven')) this.setAttribute('driven', '');
      clearTimeout(this._decay);
      // fall back to the built-in animation if levels stop arriving
      this._decay = setTimeout(() => this.removeAttribute('driven'), 400);
    }

    _apply() {
      const s = STATES[this.state] || STATES.idle;
      this.style.setProperty('--accent', s.color);
      this.style.setProperty('--ring', s.ring);
      this.setAttribute('aria-label', 'JARVIS, ' + s.label.toLowerCase());
      this.dispatchEvent(new CustomEvent('statechange', {
        bubbles: true,
        detail: { state: this.state, label: s.label, color: s.color },
      }));
    }
  }

  JarvisFace.STATES = STATES;
  if (!customElements.get('jarvis-face')) customElements.define('jarvis-face', JarvisFace);
})();
