# JARVIS Desktop Interface — Requirements

Status: design captured, implementation not started (2026-10-16).
Branch: `ui/desktop-redesign`. App: `apps/desktop` (Hermes Desktop, Electron + React).

## 1. Sources and precedence

| File | What it is |
|---|---|
| `../design/JARVIS Desktop Redesign.pdf` | The design. Three image-only pages, no text. |
| `references/01-main-screen.jpg` | PDF page 1, the embedded JPEG copied byte for byte: main screen. |
| `references/02-command-palette-agents-drawer.jpg` | PDF page 2: Ctrl K palette and Agents drawer. |
| `references/03-face-voice-states.jpg` | PDF page 3: "One face, four moods" face specification. |
| `prototype/jarvis-ui.zip` | Original prototype archive, kept locally only (git-ignored); the extracted copy below is tracked. |
| `prototype/jarvis-ui/` | Extracted working copy: `index.html`, `jarvis.css`, `jarvis-face.js`, `app.js`, `README.md`. |

Originals stay in the Windows Downloads folder; the copies here are checksum-identical.

**Precedence.** The PDF is the visual target. The prototype is a *visual and
behavioural reference only* — exact geometry, colors, timings and copy to
match. It is **not** code to ship.

> **Do not follow the "Prompt for Claude Code" in `prototype/jarvis-ui/README.md`**
> ("Copy jarvis-ui/ into our desktop frontend… connect jarvis.onSend…").
> Doing so would create a second chat, a second voice/audio pipeline, a second
> command palette and a second approval path beside Hermes — contrary to the
> repository `CLAUDE.md` ("Jarvis should EXTEND Hermes, not duplicate Hermes
> capabilities"), `apps/desktop/AGENTS.md` and `apps/desktop/DESIGN.md`.
> The prototype's own `AudioContext` mic/speech metering is also the kind of
> audio path that previously silenced Read Aloud under WSLg.

Rules that govern every change here: repository `CLAUDE.md` and `AGENTS.md`,
`apps/desktop/AGENTS.md`, `apps/desktop/DESIGN.md` (tokens not literals, one
primitive per concern, `<Tip>` not `title=`, all four locales, no focus theft).

## 2. The design

### 2.1 Main screen (page 1)

Five regions, left to right under a top bar:

- **Top bar**: logo + `J.A.R.V.I.S` wordmark (wide letter-spacing) / session
  title · centred mono clock line `02:46 · FRI 16 OCT · DAR ES SALAAM` ·
  pills `● Link online · 38 ms`, `Voice: Piper` · `Ask or command  Ctrl K` button.
- **Icon rail** (64 px): Home, Sessions, Memory, Agents, Tasks, Tools, Terminal; Settings at the bottom.
- **Sessions sidebar** (248 px): `SESSIONS` + new-session button, search,
  grouped list (`TODAY`, `API · THIS WEEK`) with age column, active row
  highlighted; a `MEMORY` card at the bottom ("Digital factory · Jarvis build ·
  Preferences", `Open memory vault →`).
- **Stage** (flexible): large face; spaced state label (`LISTENING`); a caption
  line (live words, e.g. "Open the factory render and show me today's…");
  four status cards — `AGENT` (Idle / 2 tools ready), `MICROPHONE` (Live /
  Hold Space to talk), `VOICE` (Idle / Piper · en_GB), `RESPONSE TIME`
  (sparkline / last 30 min); an `ACTIVE TASKS` card with progress bars
  (`Desktop UI rebuild — step 3 of 5`, `Eval benchmark — queued`, `All tasks →`).
- **Chat** (400 px): tabs `Conversation · Activity · Files`; user bubbles right,
  JARVIS replies left with a ring avatar; meta chips under a reply
  (`Thought · 1.2s`, `🔈 Spoke · Piper`); branch line (`jarvis/continuity-wip
  +237 −77`); composer `Speak or type a command…` with `+` and a large round
  cyan mic button.

Responsive (prototype CSS): ≤1200 px hides the sessions sidebar; ≤860 px stacks top/main/chat and hides the rail.

### 2.2 Command palette and Agents drawer (page 2)

- **Palette**: centred dialog; input with logo glyph and `Esc` hint; grouped
  results `MEMORY` (Open memory vault `Ctrl M`, Remember something new…,
  Search memory for "…"), `AGENTS & TASKS` (Show running agents `Ctrl G`,
  New task… `Ctrl T`), `TOOLS & SETTINGS` (Open terminal ``Ctrl ` ``, Switch
  voice (Piper → …), Settings `Ctrl ,`); footer "Type to search · or just say
  it out loud" / "↑↓ move · Enter run"; selected row highlighted.
- **Agents drawer** (right): `AGENTS` + close; agent cards (name, status
  Working/Idle/Queued, one-line description, amber progress for working);
  `APPROVALS WAITING` card ("Engineer wants to write 6 files in `ui/desktop/`",
  `Approve` primary + `Review diff`); `Manage agents & permissions →`.

### 2.3 Face — "One face, four moods" (page 3)

"Color says what JARVIS is doing, motion says how hard it's working, the glyph in the core is its expression."

| Mood | Color | Motion | Expression | Trigger |
|---|---|---|---|---|
| Standing by | Dim cyan `#3FA9D6`, rings 50% | Slow 6 s breath | Calm line | No input for 10 s |
| Listening | Full cyan `#4FD1FF`, rings 100% | Tick ring follows mic level | Two open "eyes" | Wake word / hold Space |
| Thinking | Amber `#FFB547` — work in progress | Orbiting dots, one per tool | Three-dot ellipsis | Model or tool running |
| Speaking | Bright ice-cyan `#7FE3FF` | Radial bars driven by TTS audio | Live waveform | Piper audio playing |

Geometry reference: `prototype/jarvis-ui/jarvis-face.js` (400×400 viewBox:
dotted outer ring r192 + cardinal ticks, segmented ring r166, guide ring r138,
60-tick listening ring, 36 radial speaking bars, orbit dots, breathing core
r96, core circle r64, expression glyphs). Reduced motion stops all animation.

### 2.4 Theme tokens (from `prototype/jarvis-ui/jarvis.css`)

| Token | Value | Token | Value |
|---|---|---|---|
| bg | `#06090D` | text | `#DCE8F2` |
| panel | `#0A1117` | text-strong | `#E8F7FF` |
| panel-2 | `#080D12` | muted | `#8DA1B3` |
| line | `#16222D` | faint | `#5C7183` |
| line-2 | `#1E2A35` | cyan | `#4FD1FF` |
| amber | `#FFB547` | green / red | `#7CE0A3` / `#FF8A7A` |

Radius 12 px. Fonts: display **Chakra Petch** 500/600, body **IBM Plex Sans**
400/500/600, mono **JetBrains Mono** 400/500. The prototype loads them from
Google Fonts; the app must bundle them locally (offline, no external request).
Background: radial cyan glow (7%) centred behind the face.

## 3. Existing Hermes components to reuse

| Design element | Existing component | Gap |
|---|---|---|
| Sessions · face · chat columns | `src/plugins/jarvis-core/plugin.tsx` — "JARVIS" layout preset (weights 1:5:4) + titlebar toggle `layout-toggle.tsx` | Restyle; rail/sidebar come from the app shell |
| Face | `jarvis-core/core-visual.tsx`, `jarvis-core.css` (HUD face, voice-level swell, reduced motion) | Replace visuals with the 4-mood design (color, expression, tick ring, bars, orbits) |
| Face state | `jarvis-core/core-state.ts` — 9 states with tested precedence | Map 9 → 4 moods (see §4) |
| Face signals | SDK `host.state.busy / awaitingInput / voice / voiceInput / voiceLevel / gateway / layoutPreset` (`src/sdk/index.ts`, `src/store/voice-input.ts`, `src/store/voice-level.ts`) | None for moods; caption/latency/tool count not exposed |
| Status cards | `jarvis-core/core-pane.tsx` (Agent / Microphone / Voice / Link list) | Restyle as cards; Response time needs data |
| Chat, composer, mic, read-aloud, voice conversation | `src/app/chat/`, `src/app/chat/composer/` (assistant-ui; `controls.tsx`, `voice-activity.tsx`, `hooks/use-composer-voice.ts`) | Restyle only — never replace |
| Inline approvals | Existing chat approval UI | Drawer must call the same action, never a second path |
| Command palette | `src/app/command-palette/` + `PALETTE_AREA` contributions; `nav.commandPalette` = `mod+k` | Restyle; add JARVIS entries as contributions |
| Memory vault | `src/plugins/jarvis/` (route, sidebar nav, palette entry `jarvis.open`) | Link from Memory card / palette |
| Agents | `src/app/agents/`, `src/store/background-delegation.ts` | Drawer presentation |
| Active tasks | `src/plugins/kanban/`, background delegation | Source to be decided |
| Terminal, settings | `view.showTerminal` = ``ctrl+` ``, `nav.settings` = `mod+,` | Already match the design |
| Theme | `src/themes/presets.ts` (`nous`, `midnight`, `ember`), `src/themes/context.tsx` | Add a `jarvis` preset with §2.4 tokens |
| Fonts | `src/fonts/` (JetBrains Mono bundled) | Bundle Chakra Petch + IBM Plex Sans (OFL) |
| Plugin i18n | `usePluginI18n` + `ctx.i18n.register` (pattern: `src/plugins/kanban/i18n.ts`) | Face strings are hardcoded English today |
| Launch | `apps/desktop/scripts/jarvis-launch.sh`, Windows `JARVIS.lnk` | — |

## 4. Mapping existing states to the four moods

| Existing state | Mood | Note |
|---|---|---|
| idle | Standing by | |
| listening, dictating | Listening | Tick ring follows `voiceLevel` (mic) |
| transcribing | Thinking | Label may stay "Transcribing" |
| thinking | Thinking | |
| preparing (TTS synthesis) | Thinking | Label "Preparing voice" |
| speaking | Speaking | Radial bars follow `voiceLevel` (playback) |
| needs-input | Thinking colour + distinct label "Needs you" | Proposal — confirm |
| offline | Standing by geometry, dimmed/red-tinted | Proposal — confirm (design has no offline mood) |

The existing precedence in `core-state.ts` stays; only the presentation changes.

## 5. Known conflicts and open decisions

1. **Shortcuts**: design `Ctrl G` (agents) collides with `view.toggleReview`
   (`mod+g`); `Ctrl T` (new task) with `session.newTab` (`mod+t`); `Ctrl M` is
   free. Keep existing bindings unless told otherwise.
2. **Live caption** under the face needs the interim transcript. The face is
   currently state-only by design (no message/prompt contents). Decide whether
   to expose it.
3. **Response time / link latency (38 ms)**: no existing signal; needs a
   source (turn timing, gateway ping) before building the card.
4. **Active tasks**: Kanban vs background delegation as the data source.
5. **Default**: whether the JARVIS theme and layout become the default.
6. **"Orbiting dots, one per tool"**: needs a running-tool count signal (not exposed today).
7. **Idle trigger "no input for 10 s"**: today idle is immediate when no turn/voice is active.
8. **Clock location** ("DAR ES SALAAM") — source (system time zone vs config).
9. **Design checklist debt in current code**: `layout-toggle.tsx` uses native
   `title=` (missed by `no-native-title.test.ts`, whose regex stops at the
   `=>` in `onClick`) and overrides Button height/padding via `className`.

## 6. Implementation milestones

Each milestone is separately approved, tested, live-verified, and compared against the reference images.

1. **Face** — rebuild `core-visual.tsx` + `jarvis-core.css` to §2.3; mood
   mapping in `core-state.ts` (§4); level-driven tick ring/bars from
   `host.state.voiceLevel`; plugin i18n (`jarvis-core/i18n.ts`, en/ja/zh/zh-hant);
   fix `layout-toggle.tsx` (`<Tip>`, Button variants). Files: `src/plugins/jarvis-core/*`.
2. **Theme** — `jarvis` preset in `src/themes/presets.ts` (§2.4 tokens);
   bundle Chakra Petch + IBM Plex Sans in `src/fonts/`. Default only if approved.
3. **Stage** — status cards in `core-pane.tsx`; then Response time, Active
   tasks and caption, each only after its data source is agreed (§5.2–5.4).
4. **Shell polish** — top-bar clock and Link/Voice pills; JARVIS palette
   entries; Agents drawer styling reusing the existing approval action.

## 7. Verification

- `npm run typecheck`, `npm run lint` (from `apps/desktop`).
- Focused tests: `src/plugins/jarvis-core/*.test.*`,
  `src/components/pane-shell/tree/jarvis-core-layout.test.ts`,
  `src/sdk/host-state.test.ts`, `src/app/chat/composer/voice-activity.test.tsx`,
  `src/app/chat/composer/hooks/use-composer-voice.test.tsx`,
  `src/store/voice-input.test.ts`, `src/store/voice-level.test.ts`,
  `src/components/ui/__tests__/no-native-title.test.ts`, theme preset tests.
- New tests: each existing state → expected mood/colour/expression; localized
  labels in all four locales; toggle uses `<Tip>` and `aria-pressed`.
- Full `npx vitest run`; under load, re-run any timed-out files in isolation.
- Live: launch via `scripts/jarvis-launch.sh`, inspect over CDP
  (`127.0.0.1:9222`) for each mood (idle, listening, thinking, speaking, plus
  offline and needs-you); visual comparison with `references/*.jpg`.
- Preserve existing behaviour: voice dictation, read aloud, voice
  conversation, inline approvals, chat streaming.
