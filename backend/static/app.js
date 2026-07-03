const API_BASE = ""; // same-origin, backend serves this file

const heroSection = document.getElementById("hero-section");
const scanningSection = document.getElementById("scanning-section");
const resultsSection = document.getElementById("results-section");
const errorMsg = document.getElementById("error-msg");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const resetBtn = document.getElementById("reset-btn");

const scanFilename = document.getElementById("scan-filename");
const scanStatus = document.getElementById("scan-status");
const scanBarFill = document.getElementById("scan-bar-fill");

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52;

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
  scanStatus.textContent = "Envoi et extraction du texte…";
  scanBarFill.style.width = "12%";

  const formData = new FormData();
  formData.append("file", file);

  let jobId;
  try {
    const res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Échec de l'envoi du fichier.");
    }
    const job = await res.json();
    jobId = job.job_id;
  } catch (e) {
    showError(e.message || "Erreur réseau lors de l'envoi.");
    return;
  }

  pollJob(jobId);
}

async function pollJob(jobId) {
  scanStatus.textContent = "Analyse statistique en cours…";
  const start = Date.now();

  const interval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status/${jobId}`);
      if (!res.ok) throw new Error("Job introuvable.");
      const job = await res.json();

      const elapsedPct = Math.min(90, 12 + (Date.now() - start) / 200);
      scanBarFill.style.width = `${Math.max(job.progress * 100, elapsedPct)}%`;

      if (job.status === "done") {
        clearInterval(interval);
        scanBarFill.style.width = "100%";
        const resultRes = await fetch(`${API_BASE}/api/result/${jobId}`);
        const result = await resultRes.json();
        renderResults(result);
      } else if (job.status === "error") {
        clearInterval(interval);
        showError(job.message || "Erreur pendant l'analyse.");
      }
    } catch (e) {
      clearInterval(interval);
      showError(e.message || "Erreur pendant l'analyse.");
    }
  }, 700);
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
  const pct = summary.overall_ai_probability;

  document.getElementById("gauge-number").textContent = Math.round(pct);
  const gaugeFill = document.getElementById("gauge-fill");
  const offset = GAUGE_CIRCUMFERENCE * (1 - pct / 100);
  gaugeFill.style.stroke = verdictColor(summary.overall_verdict);
  requestAnimationFrame(() => {
    gaugeFill.style.strokeDashoffset = offset;
  });

  document.getElementById("verdict-headline").textContent = verdictLabel(summary.overall_verdict);
  document.getElementById("stat-words").textContent = summary.word_count.toLocaleString("fr-FR");
  document.getElementById("stat-chunks").textContent = summary.total_chunks;
  document.getElementById("stat-flagged").textContent = summary.flagged_chunk_count;
  document.getElementById("stat-ppl").textContent =
    summary.mean_perplexity >= 0 ? summary.mean_perplexity : "n/d";
  document.getElementById("stat-burst").textContent = summary.mean_burstiness;

  const docView = document.getElementById("document-view");
  docView.innerHTML = "";
  chunks.forEach((chunk) => {
    const span = document.createElement("span");
    span.className = "chunk";
    span.dataset.verdict = chunk.verdict;
    span.title = `IA: ${chunk.ai_probability}% · perplexité: ${chunk.perplexity} · burstiness: ${chunk.burstiness}`;
    span.textContent = chunk.text + " ";
    docView.appendChild(span);
  });
}
