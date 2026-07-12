// PoC: render ONE labeled contest clip from WebMorseRunner's engine (headless).
// Uses its Keyer (realistic click-free CW timing) + real callsigns; we control
// the text (=label) and modulate the envelope to a tone. No browser needed.
import { DEFAULT, RunMode } from './defaults.js'
import { _setContestRef, Station } from './station.js'
import { float32ToInt16, buildWavBuffer } from './recording.js'
import { writeFileSync } from 'node:fs'

// --- headless wiring: stub the Contest + Transcript refs station.js expects ---
const stubConf = { active_contest: { exchange: [{ id: 'rst' }, { id: 'nr' }], exchange_msg: '<rst><nr>' } }
class StubContest { constructor() { this._conf = stubConf } }
const stubTst = { post: () => {} }
_setContestRef(StubContest, stubTst)

DEFAULT.RUNMODE = RunMode.Single
const RATE = DEFAULT.RATE   // 11025

// Render one text string as a keyed CW tone (Float32, mono @ RATE).
function renderCW(text, { wpm = 30, pitch = 600 } = {}) {
    const st = new Station()
    st.Wpm = wpm
    st.Amplitude = 1            // GetBlock returns the raw 0..1 envelope
    st.MsgText = ''
    st.SendText(text)           // builds the keyed envelope; MsgText = label

    const env = []
    let block
    while ((block = st.GetBlock()) !== null) env.push(...block)

    const dphi = 2 * Math.PI * pitch / RATE
    const out = new Float32Array(env.length)
    for (let i = 0; i < env.length; i++) out[i] = env[i] * Math.sin(i * dphi)
    return out
}

const label = 'K3LR 5NN 001'
const audio = renderCW(label, { wpm: 32, pitch: 620 })
const int16 = float32ToInt16(audio)
writeFileSync('poc.wav', Buffer.from(buildWavBuffer([int16], int16.length, RATE)))
console.log(`label : ${label}`)
console.log(`samples: ${audio.length}  (${(audio.length / RATE).toFixed(2)}s @ ${RATE} Hz)`)
console.log(`peak   : ${Math.max(...audio.map(Math.abs)).toFixed(3)}`)
console.log('wrote poc.wav')
