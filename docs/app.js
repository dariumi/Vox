(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");

  // ---------- вкладки и копирование ----------
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
    try { await navigator.clipboard.writeText(pre.textContent); b.textContent = "Скопировано"; } catch (e) { b.textContent = "Выделите вручную"; }
    setTimeout(() => { b.textContent = "Копировать"; }, 1600);
  }));

  // ---------- спектрограмма (палитра magma) ----------
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
    // бледные «форманты» — намёк на то, что здесь будет
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

  // ---------- WebGPU ----------
  const supported = Vox.supported();
  let gpuLabel = "";
  async function probeGpu() {
    if (!supported) {
      $("gpuName").textContent = "WebGPU недоступен"; $("gpuDot").classList.add("off");
      $("gpuNote").textContent = "нужен Chrome/Edge 113+, Firefox 141+ (Windows) или Safari 26";
      const n = $("noGpu"); n.hidden = false;
      n.textContent = "В этом браузере нет WebGPU, поэтому демо не запустится. Откройте страницу в свежем Chrome, Edge, Safari 26 или Firefox по HTTPS. Остальные разделы страницы работают без WebGPU.";
      return;
    }
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!a) throw new Error("no adapter");
      const i = a.info || {};
      gpuLabel = [i.vendor, i.architecture || i.device || i.description].filter(Boolean).join(" ") || "видеокарта";
      $("gpuName").textContent = "WebGPU доступен"; $("gpuDot").classList.add("on");
      $("gpuNote").textContent = gpuLabel;
      $("heroRun").disabled = false; $("loadTiny").disabled = false; $("loadUrl").disabled = false;
    } catch (e) {
      $("gpuName").textContent = "WebGPU есть, адаптера нет"; $("gpuDot").classList.add("off");
      $("gpuNote").textContent = "проверьте драйвер или список блокировок браузера";
      $("noGpu").hidden = false; $("noGpu").textContent = "Браузер поддерживает WebGPU, но не нашёл подходящую видеокарту. Обновите драйвер или проверьте chrome://gpu.";
    }
  }
  probeGpu();

  // ---------- демо ----------
  let model = null, voice = null, player = null, abort = null, lastWav = null;
  const setStatus = (t, err) => { const s = $("status"); s.textContent = t; s.classList.toggle("err", !!err); };
  const heroStat = (t) => { $("heroStat").textContent = t; };
  function progress(el, p) { el.hidden = false; el.firstElementChild.style.width = Math.round(p * 100) + "%"; }

  async function loadModel(src, label, tiny) {
    if (model) { try { model.dispose(); } catch (e) { /* ок */ } model = null; voice = null; }
    $("stModel").textContent = "загрузка…"; $("stModel").classList.remove("done");
    setStatus("Загружаю модель…");
    try {
      model = await Vox.load(src, {
        onProgress: (p) => progress($("loadProg"), p), onStatus: setStatus,
      });
      $("loadProg").hidden = true;
      const mb = (model.info.weightBytes / 1048576).toFixed(0);
      $("stModel").textContent = "готова"; $("stModel").classList.add("done");
      $("demoModel").textContent = label + ", " + mb + " МБ весов"; $("demoDot").classList.add("on");
      $("recBtn").disabled = false; $("refTiny").disabled = !tiny;
      setStatus("Модель загружена за " + fmt(model.info.loadMs / 1000, 1) + " с. Теперь создайте голос.");
      updateButtons();
    } catch (e) {
      $("stModel").textContent = "ошибка";
      $("loadProg").hidden = true;
      setStatus(e && e.message ? e.message : String(e), true);
    }
  }
  function updateButtons() {
    $("makeVoice").disabled = !(model && ($("refFile").files[0] || refBlob) && $("refText").value.trim());
    $("speak").disabled = !(model && voice);
  }
  let refBlob = null;
  $("loadTiny").addEventListener("click", () => loadModel("assets/tiny-f5.tmod", "тестовая модель", true));
  $("modelFile").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) loadModel(f, f.name); });
  $("loadUrl").addEventListener("click", () => { const u = $("modelUrl").value.trim(); if (u) loadModel(u, u.split("/").pop() || "модель"); });
  $("refFile").addEventListener("change", () => { refBlob = null; updateButtons(); });
  $("refText").addEventListener("input", updateButtons);
  $("refTiny").addEventListener("click", async () => {
    refBlob = await (await fetch("assets/tiny-ref.wav")).blob();
    $("refText").value = "Hi there"; $("refFile").value = "";
    setStatus("Взята запись тестовой модели (английская фраза «Hi there»)."); updateButtons();
  });
  $("speed").addEventListener("input", () => { $("speedVal").textContent = Number($("speed").value).toFixed(2).replace(/0$/, ""); });

  // запись с микрофона
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
        $("recBtn").textContent = "Записать с микрофона"; setStatus("Запись готова. Впишите ниже, что вы сказали, и создайте голос."); updateButtons();
      };
      rec.start(); $("recBtn").textContent = "Остановить запись"; setStatus("Идёт запись. Прочитайте вслух 5–12 секунд текста.");
    } catch (e) { setStatus("Нет доступа к микрофону: разрешите его в настройках сайта или загрузите файл.", true); }
  });

  $("makeVoice").addEventListener("click", async () => {
    const src = $("refFile").files[0] || refBlob;
    try {
      setStatus("Считаю спектрограмму образца…");
      const t0 = performance.now();
      voice = await model.cloneVoice(src, $("refText").value, { name: "demo" });
      $("stVoice").textContent = fmt(voice.seconds, 1) + " с"; $("stVoice").classList.add("done");
      $("demoVoice").textContent = "голос: " + fmt(voice.seconds, 1) + " с записи";
      setStatus("Голос создан за " + fmt((performance.now() - t0) / 1000, 2) + " с. Можно говорить.");
      updateButtons();
    } catch (e) { setStatus(e.message || String(e), true); }
  });

  async function synth(text, opts, ui) {
    if (abort) abort.abort();
    abort = new AbortController();
    if (player) player.stop();
    player = new Vox.StreamPlayer(model.sampleRate);
    ui.disable(true);
    try {
      const t0 = performance.now();
      const res = await model.synthesize(text, Object.assign({ voice, signal: abort.signal, onChunk: (p) => player.push(p),
        onStep: (s) => ui.progress((s.chunk + s.step / s.steps) / s.chunks) }, opts));
      ui.progress(1);
      const sp = model.spectrogram(res.pcm);
      ui.draw(sp);
      lastWav = Vox.wav(res.pcm, res.sampleRate);
      const wall = (performance.now() - t0) / 1000, st = res.stats, t = st.timing;
      ui.done(res, wall, t);
    } catch (e) {
      if (e && e.code === "aborted") ui.msg("Остановлено.");
      else ui.msg(e && e.message ? e.message : String(e), true);
    } finally { ui.disable(false); abort = null; }
  }

  $("speak").addEventListener("click", () => {
    const ui = {
      disable: (b) => { $("speak").disabled = b || !voice; $("stop").disabled = !b; },
      progress: (p) => { $("scopeBar").style.width = Math.round(p * 100) + "%"; },
      draw: (sp) => { $("scopeHint").hidden = true; drawSpec($("scope"), sp); },
      msg: (t, err) => setStatus(t, err),
      done: (res, wall, t) => {
        $("stats").hidden = false;
        $("sAudio").textContent = fmt(res.seconds, 1); $("sWall").textContent = fmt(wall, 1); $("sRtf").textContent = fmt(wall / res.seconds, 2);
        $("sDit").textContent = fmt(t.dit_ms / 1000, 1); $("sVoc").textContent = fmt(t.vocos_ms / 1000, 2);
        $("demoStat").textContent = fmt(res.seconds, 1) + " с звука за " + fmt(wall, 1) + " с";
        const a = $("dlWav"); a.hidden = false; a.href = URL.createObjectURL(lastWav);
        setStatus("Готово. Звук уже играет; скачать его можно кнопкой под спектрограммой.");
      },
    };
    setStatus("Считаю…"); $("scopeBar").style.width = "0";
    synth($("text").value, { nfe_step: Number($("nfe").value), speed: Number($("speed").value), seed: Number($("seed").value) || 1 }, ui);
  });
  $("stop").addEventListener("click", () => { if (abort) abort.abort(); if (player) player.stop(); });

  // ---------- проверка в hero: тестовая модель + запись из репозитория ----------
  $("heroRun").addEventListener("click", async () => {
    const btn = $("heroRun"); btn.disabled = true;
    const bar = $("heroBar");
    try {
      heroStat("Загружаю тестовую модель…");
      const m = await Vox.load("assets/tiny-f5.tmod");
      const blob = await (await fetch("assets/tiny-ref.wav")).blob();
      const v = await m.cloneVoice(blob, "Hi there");
      heroStat("Считаю на видеокарте…");
      const t0 = performance.now();
      const res = await m.synthesize("This is a tiny model, it only checks the pipeline.", { voice: v, nfe_step: 7, seed: 1,
        onStep: (s) => { bar.style.width = Math.round((s.chunk + s.step / s.steps) / s.chunks * 100) + "%"; } });
      const wall = (performance.now() - t0) / 1000;
      $("heroHint").hidden = true;
      drawSpec($("heroScope"), m.spectrogram(res.pcm));
      heroStat("Работает: " + fmt(res.seconds, 1) + " с звука за " + fmt(wall, 1) + " с. Для настоящего голоса загрузите свою модель в демо ниже.");
      m.dispose();
    } catch (e) { heroStat(e && e.message ? e.message : String(e)); }
    finally { btn.disabled = false; bar.style.width = "0"; }
  });
})();
