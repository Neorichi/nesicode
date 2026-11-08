/**
 * Microphone recorder → 16 kHz mono 16-bit WAV (readable by macOS speech recognition and any Whisper API).
 * Uses WebAudio rather than MediaRecorder: MediaRecorder's formats differ per engine (webm in Chrome,
 * mp4 in WebKit) and not every transcriber accepts them.
 */
export type Recording = { stop: () => Promise<Blob>; cancel: () => void; level: () => number }

const TARGET_RATE = 16000

export async function startRecording(): Promise<Recording> {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('mic-unsupported')
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
  const Ctx = window.AudioContext || (window as any).webkitAudioContext
  let ctx: AudioContext | null = null
  let source: MediaStreamAudioSourceNode
  let proc: ScriptProcessorNode
  const chunks: Float32Array[] = []
  let last = 0
  try {
    ctx = new Ctx() as AudioContext
    source = ctx.createMediaStreamSource(stream)
    proc = ctx.createScriptProcessor(4096, 1, 1)
    proc.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0)
      chunks.push(new Float32Array(data))
      let peak = 0
      for (let i = 0; i < data.length; i += 16) peak = Math.max(peak, Math.abs(data[i]))
      last = peak
    }
    source.connect(proc)
    proc.connect(ctx.destination)
  } catch (e) {
    // never leave the microphone on when the audio graph could not be built
    stream.getTracks().forEach((t) => t.stop())
    ctx?.close().catch(() => {})
    throw e
  }
  const audio = ctx

  let closed = false
  const close = () => {
    if (closed) return
    closed = true
    proc.onaudioprocess = null
    proc.disconnect()
    source.disconnect()
    stream.getTracks().forEach((t) => t.stop())
    audio.close().catch(() => {})
  }
  return {
    level: () => last,
    cancel: close,
    stop: async () => {
      const rate = audio.sampleRate
      close()
      return encodeWav(downsample(concat(chunks), rate, TARGET_RATE), TARGET_RATE)
    },
  }
}

function concat(chunks: Float32Array[]): Float32Array {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0))
  let o = 0
  for (const c of chunks) {
    out.set(c, o)
    o += c.length
  }
  return out
}

function downsample(data: Float32Array, from: number, to: number): Float32Array {
  if (from <= to) return data
  const ratio = from / to
  const out = new Float32Array(Math.floor(data.length / ratio))
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio)
    const end = Math.min(data.length, Math.floor((i + 1) * ratio))
    let sum = 0
    for (let j = start; j < end; j++) sum += data[j]
    out[i] = sum / Math.max(1, end - start)
  }
  return out
}

function encodeWav(samples: Float32Array, rate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2)
  const v = new DataView(buf)
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)))
  str(0, 'RIFF')
  v.setUint32(4, 36 + samples.length * 2, true)
  str(8, 'WAVE')
  str(12, 'fmt ')
  v.setUint32(16, 16, true)
  v.setUint16(20, 1, true) // PCM
  v.setUint16(22, 1, true) // mono
  v.setUint32(24, rate, true)
  v.setUint32(28, rate * 2, true)
  v.setUint16(32, 2, true)
  v.setUint16(34, 16, true)
  str(36, 'data')
  v.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buf], { type: 'audio/wav' })
}
