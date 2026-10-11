// ============================================================================
// Публичный объект Vox
// ============================================================================
const Vox = {
  version: VOX_VERSION,
  supported: () => typeof navigator !== "undefined" && !!navigator.gpu,
  // true — есть WebGPU; без него vox.js считает на процессоре (backend: "cpu" включает его принудительно)
  backends: () => ({ webgpu: typeof navigator !== "undefined" && !!navigator.gpu, cpu: true }),
  load: (src, opts) => VoxModel.load(src, opts),
  // Только конвертация: что угодно (HF-ссылка, URL, File, {ckpt,vocab,vocos}) → байты .tmod (можно сохранить и отдавать как файл).
  convert: (src, opts) => acquireTmod(src, opts),
  hf: { parse: parseHfRef, list: (ref, opts) => hfList(typeof ref === "string" ? parseHfRef(ref, (opts || {}).hfHost) : ref, opts || {}) },
  clearCache: cacheClear,
  config: (c) => { if (c && c.baseUrl != null) VOX_BASE_URL = c.baseUrl; return { baseUrl: VOX_BASE_URL }; },
  Voice, QVoice, VoxError, decodeAudio, wav: wavBlob, play: playPcm, StreamPlayer,
  splitText: chunkText,
  _internal: { QwenVoiceEncoder, Yielder, openQwenSource, qwenFsFromHf, qwenFsFromMap, qwenSourceToTmod, QwenTokenizer, qwenPretokenize, QwenGpu, QwenModel, QwenSession, qSample, qUniform, unpickle, readTorchArchive, parseSafetensors, convertF5, splitVocab, parseVocosPadding, melFilterbank, planFromFiles, f32ToF16Bits, f32ToBf16Bits, sniffKind, Gpu, CpuGpu, createDevice, Tmod, F5Gpu, F5Tokenizer, f5Noise, f5Timesteps, melLogSpectrogram, istftSame, resampleSinc, blen },
};
