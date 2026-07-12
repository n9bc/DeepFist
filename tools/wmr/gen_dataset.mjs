// Generate a labeled contest CW dataset from WebMorseRunner's engine (headless).
// Real callsigns + realistic keying (Keyer) + contest exchange grammar, degraded
// with QRM (interfering stations), QSB (fade), QRN (crashes), and AWGN.
// Output: <out>/clip_N.wav (6s @ 11025 Hz) + labels.jsonl ({file,text,meta}).
//
//   node gen_dataset.mjs --n 3000 --out dataset
import { DEFAULT, RunMode } from './defaults.js'
import { _setContestRef, Station } from './station.js'
import { float32ToInt16, buildWavBuffer } from './recording.js'
import { readFileSync, writeFileSync, mkdirSync, appendFileSync } from 'node:fs'

const stubConf = { active_contest: { exchange: [{ id: 'rst' }, { id: 'nr' }], exchange_msg: '<rst><nr>' } }
class StubContest { constructor() { this._conf = stubConf } }
_setContestRef(StubContest, { post: () => {} })
DEFAULT.RUNMODE = RunMode.Single

const RATE = DEFAULT.RATE               // 11025
const WIN = Math.round(6.0 * RATE)      // 6 s clips

// ---- args ----
const args = Object.fromEntries(process.argv.slice(2).join(' ').split('--').filter(Boolean)
    .map(s => s.trim().split(/\s+/)).map(([k, v]) => [k, v]))
const N = parseInt(args.n || '3000')
const OUT = args.out || 'dataset'
mkdirSync(OUT, { recursive: true })

// ---- rng (seeded, portable) ----
let _s = (parseInt(args.seed || '12345')) >>> 0
const rnd = () => (_s = (_s * 1664525 + 1013904223) >>> 0) / 4294967296
const ri = (a, b) => a + Math.floor(rnd() * (b - a))
const pick = arr => arr[ri(0, arr.length)]

const CALLS = readFileSync('calls.txt', 'utf8').split(/\r?\n/).filter(c => c.length >= 3)
const STATES = ['AL','AK','AZ','AR','CA','CO','CT','FL','GA','ID','IL','IN','IA','KS','KY',
    'LA','ME','MD','MA','MI','MN','MO','MT','NE','NV','NH','NJ','NM','NY','NC','OH','OK','OR',
    'PA','SC','TN','TX','UT','VA','WA','WI','ON','QC','BC','AB','MB','NS']
const WATTS = [5,10,25,50,100,150,200,250,300,400,500,600,750,800]

const cut = s => [...s].map(c => c === '0' ? (rnd() < 0.85 ? 'T' : 'O') : c === '9' ? 'N' : c).join('')
const report = () => rnd() < 0.75 ? '5NN' : '599'
function exchange() {
    const r = rnd()
    if (r < 0.35) { const n = ri(1, 2500); return cut(n < 1000 && rnd() < 0.8 ? String(n).padStart(3,'0') : String(n)) }
    if (r < 0.60) return cut(String(ri(1, 41)).padStart(2, '0'))          // WW zone
    if (r < 0.85) return pick(STATES)                                      // state/province
    const r2 = rnd()                                                       // power
    return r2 < 0.45 ? (rnd() < 0.5 ? 'K' : 'KW') : r2 < 0.58 ? 'QRP'
        : [...String(pick(WATTS))].map(c => c === '0' ? 'T' : c === '9' ? 'N' : c).join('')
}
function utterance() {
    const his = pick(CALLS), my = pick(CALLS), r = rnd()
    if (r < 0.22) return pick([`CQ TEST ${my} ${my}`, `CQ TEST ${my}`, `CQ CQ TEST ${my}`, `CQ ${my} TEST`])
    if (r < 0.40) return rnd() < 0.6 ? his : `${his} ${his}`
    if (r < 0.72) { const e = exchange(), rep = report(); return pick([`${his} ${rep} ${e}`, `${rep} ${e}`, `${his} ${rep} ${e} ${e}`]) }
    if (r < 0.88) return pick([`TU ${my}`, `R ${report()} ${exchange()} TU`, `${his} TU`, `TU`])
    return pick([`${his} ?`, 'NR?', 'AGN', '?', 'QRZ?'])
}

// render text -> keyed tone (Float32), placed into an n-length buffer at random offset
function renderInto(buf, text, { wpm, pitch, amp }) {
    const st = new Station(); st.Wpm = wpm; st.Amplitude = 1; st.MsgText = ''
    st.SendText(text)
    const env = []; let b; while ((b = st.GetBlock()) !== null) env.push(...b)
    const dphi = 2 * Math.PI * pitch / RATE
    const len = Math.min(env.length, buf.length)
    const start = env.length < buf.length ? ri(0, buf.length - env.length) : 0
    for (let i = 0; i < len; i++) buf[start + i] += amp * env[i] * Math.sin(i * dphi)
    return text
}

function makeClip() {
    const buf = new Float32Array(WIN)
    const wpm = ri(25, 41), pitch = 500 + rnd() * 250
    const label = renderInto(buf, utterance(), { wpm, pitch, amp: 1.0 })

    // QRM: weaker interfering stations at nearby pitches
    let nq = 0
    if (rnd() < 0.6) { nq = ri(1, 4); for (let k = 0; k < nq; k++) {
        let p = pitch + (rnd() < 0.5 ? -1 : 1) * (60 + rnd() * 300); p = Math.max(350, Math.min(950, p))
        renderInto(buf, utterance(), { wpm: ri(25, 41), pitch: p, amp: 0.2 + rnd() * 0.45 })
    }}
    // QSB
    if (rnd() < 0.5) { const rate = 0.1 + rnd() * 0.9, depth = 0.3 + rnd() * 0.6, ph = rnd() * 6.28
        for (let i = 0; i < WIN; i++) buf[i] *= 1 - depth * 0.5 * (1 - Math.cos(2*Math.PI*rate*i/RATE + ph)) }
    // QRN crashes
    if (rnd() < 0.5) { const nC = ri(0, Math.round(3 * WIN / RATE) + 1)
        for (let c = 0; c < nC; c++) { const idx = ri(0, WIN), dec = Math.round(0.01*RATE), a = 0.4 + rnd()*0.6
            for (let i = 0; i < dec && idx+i < WIN; i++) buf[idx+i] += a*(rnd()*2-1)*Math.exp(-i/(dec/4)) } }
    // AWGN at random SNR
    const snr = -6 + rnd() * 16
    let p = 0; for (let i = 0; i < WIN; i++) p += buf[i]*buf[i]; p = p/WIN + 1e-12
    const nStd = Math.sqrt(p / Math.pow(10, snr/10))
    for (let i = 0; i < WIN; i++) { const u = Math.sqrt(-2*Math.log(rnd()+1e-12))*Math.cos(6.283*rnd())
        buf[i] = Math.max(-1, Math.min(1, buf[i] + nStd*u)) }
    return { audio: buf, text: label, meta: { wpm, pitch: Math.round(pitch), snr: +snr.toFixed(1), n_qrm: nq } }
}

const t0 = Date.now()
writeFileSync(`${OUT}/labels.jsonl`, '')
for (let i = 0; i < N; i++) {
    const { audio, text, meta } = makeClip()
    const int16 = float32ToInt16(audio)
    writeFileSync(`${OUT}/clip_${i}.wav`, Buffer.from(buildWavBuffer([int16], int16.length, RATE)))
    appendFileSync(`${OUT}/labels.jsonl`, JSON.stringify({ file: `clip_${i}.wav`, text, meta }) + '\n')
    if ((i + 1) % 500 === 0) console.log(`  ${i + 1}/${N}  (${((Date.now()-t0)/1000).toFixed(0)}s)`)
}
console.log(`done: ${N} clips -> ${OUT}/  in ${((Date.now()-t0)/1000).toFixed(0)}s`)
