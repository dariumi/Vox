(function () {
  "use strict";
  const T = window.VOX_T;
  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");

  // ---------- tabs and copy buttons ----------
  document.querySelectorAll(".tabs").forEach((tabs) => {
    const btns = [...tabs.querySelectorAll("button")];
    const show = (b) => {
      btns.forEach((x) => { const on = x === b; x.setAttribute("aria-selected", on); x.tabIndex = on ? 0 : -1; $(x.getAttribute("aria-controls")).hidden = !on; });
    };
    btns.forEach((b, i) => {
      b.addEventListener("click", () => show(b));
      b.addEventListener("keydown", (e) => {
        const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (d) { const n = btns[(i + d + btns.length) % btns.length]; n.focus(); show(n); }
      });
    });
  });
  document.querySelectorAll(".copy").forEach((b) => b.addEventListener("click", async () => {
    const pre = $(b.dataset.target);
    try { await navigator.clipboard.writeText(pre.textContent); b.textContent = T.copied; } catch (e) { b.textContent = T.copyManual; }
    setTimeout(() => { b.textContent = T.copy; }, 1600);
  }));

  // ---------- spectrogram (magma palette) ----------
  const MAGMA = [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]];
  function magma(t) {
    t = Math.max(0, Math.min(1, t)) * (MAGMA.length - 1);
    const i = Math.min(MAGMA.length - 2, Math.floor(t)), f = t - i, a = MAGMA[i], b = MAGMA[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function drawIdle(cv) {
    const c = cv.getContext("2d"), w = cv.width, h = cv.height;
    const g = c.createLinearGradient(0, h, 0, 0);
    g.addColorStop(0, "#0b0618"); g.addColorStop(1, "#1a0f33");
    c.fillStyle = g; c.fillRect(0, 0, w, h);
    c.strokeStyle = "rgba(179,167,211,.12)"; c.lineWidth = 1;
    for (let i = 1; i < 8; i++) { const y = Math.round(h * i / 8) + .5; c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); }
    for (let i = 1; i < 16; i++) { const x = Math.round(w * i / 16) + .5; c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke(); }
    // faint "formants": a hint of what will appear here
    for (let k = 0; k < 4; k++) {
      const col = magma(0.35 + k * 0.17);
      c.strokeStyle = `rgba(${col[0] | 0},${col[1] | 0},${col[2] | 0},.16)`; c.lineWidth = h * (0.05 - k * 0.008);
      c.beginPath();
      for (let x = 0; x <= w; x += 8) {
        const y = h * (0.78 - k * 0.17) + Math.sin(x / w * 9 + k * 1.7) * h * 0.05 + Math.sin(x / w * 23 + k) * h * 0.02;
        if (x === 0) c.moveTo(x, y); else c.lineTo(x, y);
      }
      c.stroke();
    }
  }
  function drawSpec(cv, sp) {
    const { mel, frames, bands } = sp;
    const W = Math.min(frames, 1600), off = document.createElement("canvas");
    off.width = W; off.height = bands;
    const octx = off.getContext("2d"), img = octx.createImageData(W, bands);
    const sorted = Float32Array.from(mel).sort();
    const lo = sorted[Math.floor(sorted.length * 0.10)], hi = sorted[Math.floor(sorted.length * 0.999)], span = Math.max(1e-6, hi - lo);
    for (let x = 0; x < W; x++) {
      const t0 = Math.floor(x * frames / W), t1 = Math.max(t0 + 1, Math.floor((x + 1) * frames / W));
      for (let y = 0; y < bands; y++) {
        let v = -Infinity;
        for (let t = t0; t < t1; t++) v = Math.max(v, mel[t * bands + y]);
        const [r, g, b] = magma(Math.pow(Math.max(0, (v - lo) / span), 1.5)), o = ((bands - 1 - y) * W + x) * 4;
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    const c = cv.getContext("2d");
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = "high";
    c.drawImage(off, 0, 0, cv.width, cv.height);
  }
  drawIdle($("heroScope")); drawIdle($("scope"));

  // ---------- WebGPU or CPU ----------
  const supported = Vox.supported();
  let gpuLabel = "", hasGpu = false;
  function enableDemo() { $("heroRun").disabled = false; $("loadTiny").disabled = false; $("loadUrl").disabled = false; $("loadHf").disabled = false; }
  async function probeGpu() {
    const cpuNote = (why) => {
      $("gpuName").textContent = T.noWebgpu; $("gpuDot").classList.add("off");
      $("gpuNote").textContent = why;
      const n = $("noGpu"); n.hidden = false;
      n.textContent = T.cpuNotice;
      const sel = $("backend"); if (sel) { sel.value = "cpu"; sel.disabled = true; }
      enableDemo();
    };
    if (!supported) { cpuNote(T.needBrowsers); return; }
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!a) throw new Error("no adapter");
      const i = a.info || {};
      gpuLabel = [i.vendor, i.architecture || i.device || i.description].filter(Boolean).join(" ") || T.gpuWord;
      hasGpu = true;
      $("gpuName").textContent = T.webgpuOk; $("gpuDot").classList.add("on");
      $("gpuNote").textContent = gpuLabel;
      enableDemo();
    } catch (e) {
      cpuNote(T.noAdapter);
    }
  }
  probeGpu();

  // ---------- demo ----------
  let model = null, voice = null, player = null, abort = null, lastWav = null;
  const setStatus = (t, err) => { const s = $("status"); s.textContent = t; s.classList.toggle("err", !!err); };
  const heroStat = (t) => { $("heroStat").textContent = t; };
  function progress(el, p) { el.hidden = false; el.firstElementChild.style.width = Math.round(p * 100) + "%"; }

  let isQwen = false, needsClone = true;
  async function loadModel(src, label, tiny, extra) {
    if (model) { try { model.dispose(); } catch (e) { /* ignore */ } model = null; voice = null; }
    $("stModel").textContent = T.loading; $("stModel").classList.remove("done");
    setStatus(T.loadingModel);
    try {
      const t0 = performance.now();
      model = await Vox.load(src, Object.assign({
        backend: $("backend").value, load: $("loadMode").value,
        onProgress: (p, got, total) => {
          progress($("loadProg"), p);
          if (got && total && p < 1) {
            const sec = (performance.now() - t0) / 1000, rate = got / 1048576 / Math.max(0.1, sec), left = rate > 0 ? (total - got) / 1048576 / rate : 0;
            setStatus(T.dlProgress(fmt(got / 1048576, 0), fmt(total / 1048576, 0), fmt(rate, 1), left > 3 ? (left > 90 ? Math.round(left / 60) + " " + T.min : Math.round(left) + " " + T.sec) : ""));
          }
        },
        onStatus: setStatus,
      }, extra || {}));
      $("loadProg").hidden = true;
      isQwen = model.info.family === "qwen3-tts";
      needsClone = !isQwen || model.type === "base";
      $("qwenVoice").hidden = needsClone;
      $("f5Voice").hidden = !needsClone;
      $("qBaseOnly").hidden = !(isQwen && needsClone);
      $("refTiny").hidden = isQwen;
      for (const id of ["f5Only", "nfeBox"]) $(id).hidden = isQwen;
      $("dlVoice").hidden = true; $("stVoice").textContent = T.none; $("stVoice").classList.remove("done"); $("demoVoice").textContent = "";
      if (isQwen && !needsClone) {
        $("qSpk").innerHTML = '<option value="">' + T.noSpeaker + '</option>' + model.speakers.map((x) => `<option>${x}</option>`).join("");
        if (model.speakers.length) $("qSpk").value = model.speakers[0];
        $("qLang").innerHTML = model.languages.map((x) => `<option>${x}</option>`).join("");
        $("qIns").placeholder = model.capabilities.requires_instruct ? T.insRequired : T.insOptional;
        $("qSpk").disabled = !model.capabilities.named_speakers; $("qIns").disabled = !model.capabilities.instruct;
        $("stQVoice").textContent = T.chosenOnRun;
      }
      const mb = (model.info.weightBytes / 1048576).toFixed(0);
      $("stModel").textContent = T.ready; $("stModel").classList.add("done");
      $("demoModel").textContent = T.weightsOf(label, mb); $("demoDot").classList.add("on");
      $("recBtn").disabled = false; $("refTiny").disabled = !tiny;
      const nt = model.info.net, bk = model.info.backend === "cpu" ? T.cpuWord : "WebGPU";
      setStatus(T.loadedIn(fmt(model.info.loadMs / 1000, 1), bk, nt ? T.netInfo(nt.requests, fmt(nt.mib, 0)) : "") + " " + (isQwen ? T.pickSpeaker : T.makeVoiceNow));
      updateButtons();
    } catch (e) {
      $("stModel").textContent = T.error;
      $("loadProg").hidden = true;
      setStatus(e && e.message ? e.message : String(e), true);
    }
  }
  function updateButtons() {
    $("makeVoice").disabled = !(model && ($("refFile").files[0] || refBlob) && $("refText").value.trim());
    const ready = !!(model && (!needsClone || voice));
    $("speak").disabled = !ready;
    $("liveStart").disabled = !ready || !!duplex;
  }
  // Qwen3-TTS voice from the selected speaker / description / .tvoice file
  async function qwenVoice() {
    return model.speakerVoice($("qSpk").value, $("qIns").value.trim(), { language: $("qLang").value });
  }
  async function currentVoice() { return needsClone ? voice : qwenVoice(); }
  let refBlob = null;
  $("loadTiny").addEventListener("click", () => loadModel("assets/tiny-f5.tmod", T.tinyModel, true));
  $("loadHf").addEventListener("click", () => {
    const u = $("hfLink").value.trim();
    if (u) loadModel(u, u.replace(/^https?:\/\/huggingface\.co\//, "").replace(/^hf:/, ""), false, { dtype: $("hfDtype").value, quant: $("hfQuant").value });
  });
  $("modelFile").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) loadModel(f, f.name); });
  $("loadUrl").addEventListener("click", () => { const u = $("modelUrl").value.trim(); if (u) loadModel(u, u.split("/").pop() || T.modelWord); });
  $("refFile").addEventListener("change", () => { refBlob = null; updateButtons(); });
  $("refText").addEventListener("input", updateButtons);
  $("refTiny").addEventListener("click", async () => {
    refBlob = await (await fetch("assets/tiny-ref.wav")).blob();
    $("refText").value = "Hi there"; $("refFile").value = "";
    setStatus(T.tinyRefTaken); updateButtons();
  });
  $("speed").addEventListener("input", () => { $("speedVal").textContent = Number($("speed").value).toFixed(2).replace(/0$/, ""); });

  // microphone recording
  let rec = null;
  $("recBtn").addEventListener("click", async () => {
    if (rec) { rec.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
      const chunks = [];
      rec = new MediaRecorder(stream);
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        refBlob = new Blob(chunks, { type: rec.mimeType }); rec = null; $("refFile").value = "";
        $("recBtn").textContent = T.rec; setStatus(T.recReady); updateButtons();
      };
      rec.start(); $("recBtn").textContent = T.recStop; setStatus(T.recGoing);
    } catch (e) { setStatus(T.noMic, true); }
  });

  function showVoice(v, label) {
    voice = v;
    $("stVoice").textContent = label; $("stVoice").classList.add("done");
    $("demoVoice").textContent = T.voiceLabel(label);
    if (isQwen && v.toTvoice) {
      const a = $("dlVoice"); if (a.href && a.href.startsWith("blob:")) URL.revokeObjectURL(a.href);
      a.href = URL.createObjectURL(new Blob([v.toTvoice()], { type: "application/octet-stream" })); a.hidden = false;
    }
    updateButtons();
  }
  $("qTvoice").addEventListener("change", async () => {
    const f = $("qTvoice").files[0];
    if (!f || !model) return;
    try { const v = model.voiceFromFile(new Uint8Array(await f.arrayBuffer())); showVoice(v, f.name); setStatus(T.voiceFileOk); }
    catch (e) { setStatus(e.message || String(e), true); }
  });
  $("makeVoice").addEventListener("click", async () => {
    const src = $("refFile").files[0] || refBlob;
    const btn = $("makeVoice"); btn.disabled = true;
    try {
      setStatus(isQwen ? T.voiceQwen : T.voiceF5);
      const t0 = performance.now();
      const v = await model.cloneVoice(src, $("refText").value, { name: "demo", xvectorOnly: isQwen && $("xvOnly").checked,
        onProgress: (p) => progress($("voiceProg"), p) });
      $("voiceProg").hidden = true;
      showVoice(v, T.secOfRec(fmt(v.seconds, 1)));
      setStatus(T.voiceMade(fmt((performance.now() - t0) / 1000, 1)));
    } catch (e) { $("voiceProg").hidden = true; setStatus(e.message || String(e), true); }
    finally { updateButtons(); }
  });

  function setPlayer(blob) {
    const au = $("player");
    if (au.src && au.src.startsWith("blob:")) URL.revokeObjectURL(au.src);
    au.src = URL.createObjectURL(blob); au.hidden = false;
  }
  async function synth(text, opts, ui) {
    if (abort) abort.abort();
    let v;
    try { v = await currentVoice(); } catch (e) { ui.msg(e && e.message ? e.message : String(e), true); return; }
    abort = new AbortController();
    if (player) player.stop();
    player = new Vox.StreamPlayer(model.sampleRate);
    ui.disable(true);
    try {
      const t0 = performance.now();
      const res = await model.synthesize(text, Object.assign({ voice: v, signal: abort.signal, onChunk: (p) => player.push(p),
        onStep: (s) => ui.progress((s.chunk + s.step / s.steps) / s.chunks) }, opts));
      ui.progress(1);
      const sp = model.spectrogram(res.pcm);
      ui.draw(sp);
      lastWav = Vox.wav(res.pcm, res.sampleRate);
      const wall = (performance.now() - t0) / 1000, st = res.stats, t = st.timing;
      ui.done(res, wall, t);
    } catch (e) {
      if (e && e.code === "aborted") ui.msg(T.stopped);
      else ui.msg(e && e.message ? e.message : String(e), true);
    } finally { ui.disable(false); abort = null; }
  }

  $("speak").addEventListener("click", () => {
    const ui = {
      disable: (b) => { $("speak").disabled = b || !(!needsClone || voice); $("stop").disabled = !b; },
      progress: (p) => { $("scopeBar").style.width = Math.round(p * 100) + "%"; },
      draw: (sp) => { $("scopeHint").hidden = true; drawSpec($("scope"), sp); },
      msg: (t, err) => setStatus(t, err),
      done: (res, wall, t) => {
        $("stats").hidden = false;
        $("sAudio").textContent = fmt(res.seconds, 1); $("sWall").textContent = fmt(wall, 1); $("sRtf").textContent = fmt(wall / res.seconds, 2);
        $("sDit").textContent = fmt(t.dit_ms / 1000, 1); $("sVoc").textContent = fmt(t.vocos_ms / 1000, 2);
        $("demoStat").textContent = T.audioFor(fmt(res.seconds, 1), fmt(wall, 1));
        const a = $("dlWav"); a.hidden = false; a.href = URL.createObjectURL(lastWav);
        setPlayer(lastWav);
        setStatus(T.donePlaying);
      },
    };
    setStatus(T.computing); $("scopeBar").style.width = "0";
    synth($("text").value, isQwen ? { seed: Number($("seed").value) || 1, language: $("qLang").value }
      : { nfe_step: Number($("nfe").value), speed: Number($("speed").value), seed: Number($("seed").value) || 1 }, ui);
  });
  $("stop").addEventListener("click", () => { if (abort) abort.abort(); if (player) player.stop(); });

  // ---------- live input (duplex streaming) ----------
  let duplex = null, livePlayer = null, liveTimer = null;
  const liveSay = (t, err) => { const s = $("liveStatus"); s.textContent = t; s.classList.toggle("err", !!err); };
  function liveStats() {
    if (!duplex) return;
    const st = duplex.stats, inf = duplex.info();
    $("lvFirst").textContent = st.firstAudioMs != null ? fmt(st.firstAudioMs / 1000, 2) : "–";
    $("lvAudio").textContent = fmt(st.audioSamples / model.sampleRate, 1);
    $("lvSpoken").textContent = String(inf.committed_bytes != null ? inf.committed_bytes : 0);
  }
  $("liveStart").addEventListener("click", async () => {
    let v;
    try { v = await currentVoice(); } catch (e) { liveSay(e && e.message ? e.message : String(e), true); return; }
    if (player) player.stop();
    livePlayer = new Vox.StreamPlayer(model.sampleRate);
    const rec = [];
    const d = duplex = model.duplex({ voice: v, commit: $("liveCommit").value, idle_ms: Number($("liveIdle").value) || 0,
      utterance_gap_ms: isQwen ? 1500 : 0, seed: Number($("seed").value) || 1, language: isQwen ? $("qLang").value : undefined });
    $("liveText").disabled = false; $("liveText").value = ""; $("liveText").focus();
    $("liveStart").disabled = true; $("liveFlush").disabled = false; $("liveStop").disabled = false;
    $("stLive").textContent = T.listening; $("stLive").classList.add("done");
    liveSay(d.caps && d.caps.text_in === "token" ? T.liveTokens : T.liveChunks);
    liveTimer = setInterval(liveStats, 300);
    (async () => {
      try { for await (const c of d.audio()) { livePlayer.push(c); rec.push(c); } liveSay(T.liveDone); if (rec.length) { const n = rec.reduce((s2, q) => s2 + q.length, 0), all = new Float32Array(n); let o = 0; for (const q of rec) { all.set(q, o); o += q.length; } lastWav = Vox.wav(all, model.sampleRate); setPlayer(lastWav); const a = $("dlWav"); a.hidden = false; a.href = URL.createObjectURL(lastWav); } }
      catch (e) { if (!(e && e.code === "aborted")) liveSay(e && e.message ? e.message : String(e), true); }
      finally { clearInterval(liveTimer); liveStats(); duplex = null; $("liveText").disabled = true; $("liveFlush").disabled = true; $("liveStop").disabled = true;
        $("stLive").textContent = T.liveStopped; $("stLive").classList.remove("done"); updateButtons(); }
    })();
  });
  $("liveText").addEventListener("input", () => {
    if (!duplex) return;
    try { duplex.setText($("liveText").value); } catch (e) { liveSay(e && e.message ? e.message : String(e), true); }
  });
  $("liveFlush").addEventListener("click", () => { if (duplex) duplex.flush(); });
  $("liveStop").addEventListener("click", () => { if (duplex) { duplex.end(); liveSay(T.liveEnding); $("liveText").disabled = true; $("liveStop").disabled = true; } });

  // ---------- hero check: tiny model + reference recording from the repo ----------
  $("heroRun").addEventListener("click", async () => {
    const btn = $("heroRun"); btn.disabled = true;
    const bar = $("heroBar");
    try {
      heroStat(T.heroLoading);
      const m = await Vox.load("assets/tiny-f5.tmod");
      const blob = await (await fetch("assets/tiny-ref.wav")).blob();
      const v = await m.cloneVoice(blob, "Hi there");
      heroStat(T.heroComputing);
      const t0 = performance.now();
      const res = await m.synthesize("This is a tiny model, it only checks the pipeline.", { voice: v, nfe_step: 7, seed: 1,
        onStep: (s) => { bar.style.width = Math.round((s.chunk + s.step / s.steps) / s.chunks * 100) + "%"; } });
      const wall = (performance.now() - t0) / 1000;
      $("heroHint").hidden = true;
      drawSpec($("heroScope"), m.spectrogram(res.pcm));
      heroStat(T.heroDone(fmt(res.seconds, 1), fmt(wall, 1)));
      m.dispose();
    } catch (e) { heroStat(e && e.message ? e.message : String(e)); }
    finally { btn.disabled = false; bar.style.width = "0"; }
  });
})();
