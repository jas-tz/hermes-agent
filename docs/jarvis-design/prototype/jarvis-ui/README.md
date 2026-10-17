# JARVIS desktop UI

Plain HTML/CSS/JS, no framework and no build step. It works in Electron, Tauri, a browser, or inside your existing Jarvis web UI.

| File | What it is |
|---|---|
| `index.html` | The full desktop layout: top bar, icon rail, sessions, face, status cards, chat, Ctrl K palette, agents drawer |
| `jarvis.css` | Theme (colors, fonts) + layout. Change colors in `:root` at the top |
| `jarvis-face.js` | The animated face as `<jarvis-face>`. You can use it on its own anywhere |
| `app.js` | Wiring + the `window.jarvis` API your backend calls |

## Try it

Open `index.html` in a browser.

- Type a message and press Enter to see the demo go thinking → speaking → idle.
- Hold **Space** (or the mic button) to see listening. It uses your real mic if you allow it.
- **Ctrl K** opens the palette. **Ctrl G** opens agents. **Esc** closes.

## Connect it to Jarvis

In `app.js`, set `DEMO = false`, then from your own code:

```js
// user typed something
jarvis.onSend = async (text) => {
  jarvis.setState('thinking');
  const reply = await askJarvis(text);          // your backend call
  jarvis.addMessage('jarvis', reply.text);
  if (reply.audio) await jarvis.playSpeech(reply.audio, reply.text);  // Piper wav → face speaks
  else jarvis.setState('idle');
};

// push-to-talk released
jarvis.onPushToTalkEnd = () => { /* send recorded audio to speech-to-text, then call jarvis.onSend(text) */ };

// palette / rail buttons
jarvis.onCommand = (name) => { /* 'memory' | 'tasks' | 'terminal' | 'voice' | 'settings' | 'new-session' … */ };
```

Map your backend events to face states:

| Jarvis event | Call |
|---|---|
| Wake word heard / mic on | `jarvis.startMic()` (sets *listening*, face follows your voice) |
| Request sent, model or tool running | `jarvis.setState('thinking', 'Reading files…')` |
| Piper audio ready | `jarvis.playSpeech(wavBlobOrUrl, text)` |
| Done | `jarvis.setState('idle')` |
| Link down | `jarvis.setStatus({ link: 'Link offline' })` |

## Prompt for Claude Code

```
Copy jarvis-ui/ into our desktop frontend. Keep index.html's layout,
jarvis.css and jarvis-face.js as they are. In app.js set DEMO=false
and connect jarvis.onSend, onPushToTalkEnd and onCommand to our
existing backend (agent, Piper TTS, sessions, memory). Do one hook
at a time and show me how to test each.
```
