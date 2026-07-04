// Scriptoire — 100% client-side AI-text detector.
// No server calls: extraction, heuristics and the perplexity model all run
// in the browser. Everything below is a direct JS port of the Python
// version's logic (chunking.py / heuristics.py / perplexity.py / scoring.py).

import { AutoTokenizer, AutoModelForCausalLM, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.5.136/build/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.5.136/build/pdf.worker.min.mjs";

// mammoth.js (docx -> text) attaches a global `mammoth` object
const MAMMOTH_SRC = "https://cdn.jsdelivr.net/npm/mammoth@1.7.2/mammoth.browser.min.js";
function loadMammoth() {
  return new Promise((resolve, reject) => {
    if (window.mammoth) return resolve(window.mammoth);
    const s = document.createElement("script");
    s.src = MAMMOTH_SRC;
    s.onload = () => resolve(window.mammoth);
    s.onerror = () => reject(new Error("Impossible de charger mammoth.js"));
    document.head.appendChild(s);
  });
}

env.allowLocalModels = false; // always fetch from the HF CDN, never look for local files

const MODEL_ID = "Xenova/distilgpt2";
const TARGET_WORDS_PER_CHUNK = 150;
const MIN_WORDS_PER_CHUNK = 60;
const MAX_TOKENS = 512;

const PPL_LOW = 12.0;
const PPL_HIGH = 55.0;
const BURSTINESS_LOW = 0.35;
const BURSTINESS_HIGH = 0.85;
const WEIGHT_PERPLEXITY = 0.55;
const WEIGHT_BURSTINESS = 0.25;
const WEIGHT_REPETITION = 0.20;

// ---------- Text extraction ----------

async function extractText(file) {
  const ext = file.name.toLowerCase().split(".").pop();
  if (ext === "txt") {
    return await file.text();
  }
  if (ext === "pdf") {
    return await extractPdf(file);
  }
  if (ext === "docx") {
    return await extractDocx(file);
  }
  throw new Error(`Format non supporté : .${ext}`);
}

async function extractPdf(file) {
  const buf = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  const parts = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map((it) => it.str).join(" ");
    parts.push(pageText);
    setScanStatus(`Extraction du PDF… page ${i}/${pdf.numPages}`);
    setScanProgress(0.05 + 0.15 * (i / pdf.numPages));
  }
  return parts.join("\n\n");
}

async function extractDocx(file) {
  const mammoth = await loadMammoth();
  const buf = await file.arrayBuffer();
  const result = await mammoth.extractRawText({ arrayBuffer: buf });
  return result.value;
}

// ---------- Chunking (port of chunking.py) ----------

const SENTENCE_SPLIT = /(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9"'])/;

function splitSentences(text) {
  text = text.trim();
  if (!text) return [];
  return text.split(SENTENCE_SPLIT).map((s) => s.trim()).filter(Boolean);
}

function chunkDocument(text) {
  const chunks = [];
  let cursor = 0;
  let bufferSentences = [];
  let bufferWords = 0;
  let bufferStart = null;

  const sentences = splitSentences(text);

  for (const sentence of sentences) {
    let idx = text.indexOf(sentence, cursor);
    if (idx === -1) idx = cursor;
    const endIdx = idx + sentence.length;

    if (bufferStart === null) bufferStart = idx;

    bufferSentences.push(sentence);
    bufferWords += sentence.split(/\s+/).filter(Boolean).length;
    cursor = endIdx;

    if (bufferWords >= TARGET_WORDS_PER_CHUNK) {
      chunks.push({ text: bufferSentences.join(" "), start: bufferStart, end: endIdx });
      bufferSentences = [];
      bufferWords = 0;
      bufferStart = null;
    }
  }

  if (bufferSentences.length) {
    const chunkText = bufferSentences.join(" ");
    if (chunks.length && bufferWords < MIN_WORDS_PER_CHUNK) {
      const last = chunks.pop();
      chunks.push({ text: last.text + " " + chunkText, start: last.start, end: cursor });
    } else {
      chunks.push({ text: chunkText, start: bufferStart || 0, end: cursor });
    }
  }

  return chunks;
}

// ---------- Heuristics (port of heuristics.py) ----------

const WORD_RE = /[A-Za-zÀ-ÿ']+/g;

function words(text) {
  return (text.toLowerCase().match(WORD_RE)) || [];
}

function burstiness(text) {
  const sentences = splitSentences(text);
  const lengths = sentences.map((s) => (s.match(WORD_RE) || []).length).filter((n) => n > 0);
  if (lengths.length < 2) return 1.0;

  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  if (mean === 0) return 1.0;
  const variance = lengths.reduce((a, b) => a + (b - mean) ** 2, 0) / lengths.length;
  const stdev = Math.sqrt(variance);
  return stdev / mean;
}

function repetitionScore(text) {
  const w = words(text);
  if (w.length < 8) return 0.0;

  const counts = new Map();
  for (let i = 0; i < w.length - 2; i++) {
    const tri = w[i] + " " + w[i + 1] + " " + w[i + 2];
    counts.set(tri, (counts.get(tri) || 0) + 1);
  }
  let repeated = 0;
  let total = 0;
  for (const c of counts.values()) {
    total += c;
    if (c > 1) repeated += c;
  }
  return Math.min(1.0, repeated / Math.max(1, w.length - 2));
}

// ---------- Perplexity via in-browser model (port of perplexity.py) ----------

let tokenizer = null;
let model = null;
let modelReady = false;

async function loadModel(onProgress) {
  try {
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: onProgress });
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { progress_callback: onProgress });
    modelReady = true;
  } catch (e) {
    console.warn("Modèle de perplexité indisponible, mode dégradé (heuristiques seules) :", e);
    modelReady = false;
  }
}

async function computePerplexity(text) {
  if (!modelReady) return null;
  try {
    const encoded = await tokenizer(text, { truncation: true, max_length: MAX_TOKENS });
    const inputIds = encoded.input_ids;
    const seqLen = inputIds.dims[1];
    if (seqLen < 2) return null;

    const output = await model({ input_ids: inputIds, attention_mask: encoded.attention_mask });
    const logits = output.logits; // [1, seqLen, vocabSize]
    const vocabSize = logits.dims[2];
    const logitsData = logits.data;
    const idsData = inputIds.data;

    let totalNLL = 0;
    let count = 0;
    for (let t = 0; t < seqLen - 1; t++) {
      const targetId = Number(idsData[t + 1]);
      const offset = t * vocabSize;
      let maxLogit = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        const val = logitsData[offset + v];
        if (val > maxLogit) maxLogit = val;
      }
      let sumExp = 0;
      for (let v = 0; v < vocabSize; v++) {
        sumExp += Math.exp(logitsData[offset + v] - maxLogit);
      }
      const logProbTarget = logitsData[offset + targetId] - maxLogit - Math.log(sumExp);
      totalNLL += -logProbTarget;
      count++;
    }
    const meanNLL = totalNLL / count;
    return Math.exp(meanNLL);
  } catch (e) {
    console.warn("Erreur de calcul de perplexité sur ce segment :", e);
    return null;
  }
}

// ---------- Scoring (port of scoring.py) ----------

function clip01(x) {
  return Math.max(0, Math.min(1, x));
}

function perplexityToAiScore(ppl) {
  if (ppl === null || ppl === undefined) return 0.5;
  if (ppl <= PPL_LOW) return 1.0;
  if (ppl >= PPL_HIGH) return 0.0;
  return clip01(1.0 - (ppl - PPL_LOW) / (PPL_HIGH - PPL_LOW));
}

function burstinessToAiScore(b) {
  if (b <= BURSTINESS_LOW) return 1.0;
  if (b >= BURSTINESS_HIGH) return 0.0;
  return clip01(1.0 - (b - BURSTINESS_LOW) / (BURSTINESS_HIGH - BURSTINESS_LOW));
}

function verdictFromScore(aiProbability) {
  if (aiProbability >= 70) return "ai";
  if (aiProbability >= 40) return "mixed";
  return "human";
}

async function scoreChunk(index, rawText, start, end) {
  const ppl = await computePerplexity(rawText);
  const b = burstiness(rawText);
  const rep = repetitionScore(rawText);

  const pplScore = perplexityToAiScore(ppl);
  const burstScore = burstinessToAiScore(b);
  const repScore = clip01(rep);

  const combined =
    WEIGHT_PERPLEXITY * pplScore + WEIGHT_BURSTINESS * burstScore + WEIGHT_REPETITION * repScore;
  const aiProbability = Math.round(combined * 1000) / 10;

  return {
    index,
    text: rawText,
    start,
    end,
    perplexity: ppl !== null ? Math.round(ppl * 100) / 100 : -1,
    burstiness: Math.round(b * 1000) / 1000,
    repetitionScore: Math.round(rep * 1000) / 1000,
    aiProbability,
    verdict: verdictFromScore(aiProbability),
  };
}

async function analyzeDocument(text, onProgress) {
  const rawChunks = chunkDocument(text);
  const results = [];

  for (let i = 0; i < rawChunks.length; i++) {
    const c = rawChunks[i];
    results.push(await scoreChunk(i, c.text, c.start, c.end));

    if (onProgress) onProgress(i + 1, rawChunks.length);
    // yield to the UI thread every few chunks so the page never freezes
    if (i % 3 === 0) await new Promise((r) => setTimeout(r, 0));
  }

  let meanPpl = 0, meanBurst = 0, overallAi = 0, flagged = 0;
  const withPpl = results.filter((c) => c.perplexity >= 0);
  if (withPpl.length) meanPpl = withPpl.reduce((a, c) => a + c.perplexity, 0) / withPpl.length;
  if (results.length) {
    meanBurst = results.reduce((a, c) => a + c.burstiness, 0) / results.length;
    overallAi = results.reduce((a, c) => a + c.aiProbability, 0) / results.length;
    flagged = results.filter((c) => c.verdict === "ai").length;
  }

  return {
    summary: {
      totalChars: text.length,
      totalChunks: results.length,
      wordCount: text.split(/\s+/).filter(Boolean).length,
      overallAiProbability: Math.round(overallAi * 10) / 10,
      overallVerdict: verdictFromScore(overallAi),
      meanPerplexity: Math.round(meanPpl * 100) / 100,
      meanBurstiness: Math.round(meanBurst * 1000) / 1000,
      flaggedChunkCount: flagged,
    },
    chunks: results,
  };
}

// ---------- UI wiring ----------

const heroSection = document.getElementById("hero-section");
const scanningSection = document.getElementById("scanning-section");
const resultsSection = document.getElementById("results-section");
const errorMsg = document.getElementById("error-msg");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const resetBtn = document.getElementById("reset-btn");

const scanFilename = document.getElementById("scan-filename");
const scanStatusEl = document.getElementById("scan-status");
const scanBarFill = document.getElementById("scan-bar-fill");

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52;
let modelLoadStarted = false;

function setScanStatus(msg) {
  scanStatusEl.textContent = msg;
}
function setScanProgress(pct) {
  scanBarFill.style.width = `${Math.round(pct * 100)}%`;
}

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

resetBtn.addEventListener("click", () => {
  resultsSection.classList.add("hidden");
  heroSection.classList.remove("hidden");
  fileInput.value = "";
});

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove("hidden");
  scanningSection.classList.add("hidden");
  heroSection.classList.remove("hidden");
}

async function handleFile(file) {
  errorMsg.classList.add("hidden");
  heroSection.classList.add("hidden");
  scanningSection.classList.remove("hidden");
  scanFilename.textContent = file.name;
  setScanProgress(0.02);

  if (!modelLoadStarted) {
    modelLoadStarted = true;
    setScanStatus("Chargement du modèle local (première fois seulement)…");
    await loadModel((p) => {
      if (p.status === "progress") {
        setScanProgress(0.03 + 0.02 * (p.progress || 0));
      }
    });
  }

  let text;
  try {
    setScanStatus("Extraction du texte…");
    text = await extractText(file);
  } catch (e) {
    showError(e.message || "Échec de l'extraction du texte.");
    return;
  }

  if (!text || !text.trim()) {
    showError("Aucun texte extrait du document (scan sans OCR ?)");
    return;
  }

  setScanStatus("Analyse statistique en cours…");
  try {
    const result = await analyzeDocument(text, (done, total) => {
      setScanProgress(0.1 + 0.9 * (done / total));
      setScanStatus(`Analyse en cours… segment ${done}/${total}`);
    });
    renderResults(result);
  } catch (e) {
    showError(e.message || "Erreur pendant l'analyse.");
  }
}

function verdictLabel(v) {
  if (v === "ai") return "Probablement généré par IA";
  if (v === "mixed") return "Résultat incertain";
  return "Probablement écrit par un humain";
}

function verdictColor(v) {
  if (v === "ai") return "#E8A33D";
  if (v === "mixed") return "#A78BFA";
  return "#4FC3B0";
}

function renderResults(result) {
  scanningSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  const { summary, chunks } = result;
  const pct = summary.overallAiProbability;

  document.getElementById("gauge-number").textContent = Math.round(pct);
  const gaugeFill = document.getElementById("gauge-fill");
  const offset = GAUGE_CIRCUMFERENCE * (1 - pct / 100);
  gaugeFill.style.stroke = verdictColor(summary.overallVerdict);
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = offset;
  });

  document.getElementById("verdict-headline").textContent = verdictLabel(summary.overallVerdict);
  document.getElementById("stat-words").textContent = summary.wordCount.toLocaleString("fr-FR");
  document.getElementById("stat-chunks").textContent = summary.totalChunks;
  document.getElementById("stat-flagged").textContent = summary.flaggedChunkCount;
  document.getElementById("stat-ppl").textContent =
    summary.meanPerplexity >= 0 ? summary.meanPerplexity : "n/d";
  document.getElementById("stat-burst").textContent = summary.meanBurstiness;

  const docView = document.getElementById("document-view");
  docView.innerHTML = "";
  chunks.forEach((chunk) => {
    const span = document.createElement("span");
    span.className = "chunk";
    span.dataset.verdict = chunk.verdict;
    span.title = `IA: ${chunk.aiProbability}% · perplexité: ${chunk.perplexity} · burstiness: ${chunk.burstiness}`;
    span.textContent = chunk.text + " ";
    docView.appendChild(span);
  });
}
