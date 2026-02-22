const statusEl = document.getElementById("status");
const docList = document.getElementById("docList");
const deleteAllBtn = document.getElementById("deleteAllBtn");
const uploadBtn = document.getElementById("uploadBtn");
const uploadMsg = document.getElementById("uploadMsg");
const fileInput = document.getElementById("fileInput");
const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const includeDocSummaries = document.getElementById("includeDocSummaries");
const selectAllDocs = document.getElementById("selectAllDocs");
const scopeSummary = document.getElementById("scopeSummary");

let docsCache = [];
let selectedDocIds = new Set();
let scopeMode = "all";

function isDocReady(doc) {
  return String(doc.status || "").toLowerCase() === "ready";
}

function getReadyDocIds(docs) {
  return docs.filter(isDocReady).map((doc) => doc.id);
}

function syncSelectionToDocs(docs) {
  const readyDocIds = getReadyDocIds(docs);
  const readySet = new Set(readyDocIds);
  if (scopeMode === "all") {
    selectedDocIds = new Set(readyDocIds);
    return;
  }
  selectedDocIds = new Set([...selectedDocIds].filter((docId) => readySet.has(docId)));
}

function updateScopeControls(docs) {
  const readyDocIds = getReadyDocIds(docs);
  const selectedCount = readyDocIds.filter((docId) => selectedDocIds.has(docId)).length;
  scopeSummary.textContent = `${selectedCount}/${readyDocIds.length} selected`;

  selectAllDocs.disabled = readyDocIds.length === 0;
  selectAllDocs.indeterminate = selectedCount > 0 && selectedCount < readyDocIds.length;
  selectAllDocs.checked = readyDocIds.length > 0 && selectedCount === readyDocIds.length;
}

async function fetchHealth() {
  const res = await fetch("/api/health");
  const data = await res.json();
  statusEl.textContent = data.vlm_enabled ? "VLM enabled" : "VLM disabled";
}

function renderDocs(docs) {
  docList.innerHTML = "";
  if (!docs.length) {
    const li = document.createElement("li");
    li.textContent = "No documents yet.";
    docList.appendChild(li);
    updateScopeControls(docs);
    return;
  }
  docs.forEach((doc) => {
    const ready = isDocReady(doc);
    const checked = ready && selectedDocIds.has(doc.id);
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `
      <div class="doc-main-wrap">
        <label class="doc-scope-toggle" title="${ready ? "Include this document in chat context" : "Only ready documents can be selected"}">
          <input class="doc-select-checkbox" data-doc-id="${doc.id}" type="checkbox" ${checked ? "checked" : ""} ${ready ? "" : "disabled"} />
        </label>
        <div class="doc-main">
          <strong>${doc.filename}</strong>
          <span>${doc.status}${ready ? "" : " (not selectable yet)"}</span>
        </div>
      </div>
      <button class="delete-btn" data-doc-id="${doc.id}" type="button">Delete</button>
    `;
    docList.appendChild(li);
  });
  updateScopeControls(docs);
}

async function fetchDocs() {
  const res = await fetch("/api/documents");
  const data = await res.json();
  docsCache = data.documents || [];
  syncSelectionToDocs(docsCache);
  renderDocs(docsCache);
}

function addMessage(text, role = "assistant") {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `<div class="meta">${role}</div><div>${text}</div>`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatAnswerParagraphs(text) {
  const normalized = String(text || "").replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return "<p>I could not generate an answer.</p>";
  }
  const paragraphs = normalized
    .split(/\n\s*\n+/)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => p.replace(/^#{1,6}\s*/gm, "").trim());
  return paragraphs
    .map((p) => `<p>${escapeHtml(p).replace(/\n/g, "<br />")}</p>`)
    .join("");
}

function formatSources(sources) {
  const unique = [];
  const seenDocIds = new Set();
  for (const s of sources || []) {
    const docId = String(s.doc_id || "").trim();
    if (!docId || seenDocIds.has(docId)) continue;
    seenDocIds.add(docId);
    unique.push(s);
  }
  if (!unique.length) {
    return "<p class=\"source-empty\">No sources returned.</p>";
  }
  const lines = unique.map((s) => {
    const name = s.doc_filename || "unknown";
    const page = s.page ?? "?";
    const excerpt = String(s.content || "").slice(0, 180);
    return `<p class="source-item"><strong>${escapeHtml(name)}</strong> (${escapeHtml(s.doc_id)}:${escapeHtml(page)})<br />${escapeHtml(excerpt)}...</p>`;
  });
  return lines.join("");
}

function renderAssistantResponse(answer, sources, includeSummaries) {
  const answerHtml = formatAnswerParagraphs(answer);
  const sourcesHtml = formatSources(sources || []);
  const sourcesTitle = includeSummaries ? "Document Summaries / Sources" : "Sources";
  return (
    `<div class="answer-body">${answerHtml}</div>` +
    `<div class="sources-body"><div class="sources-title">${sourcesTitle}</div>${sourcesHtml}</div>`
  );
}

async function uploadFiles(files) {
  if (!files || files.length === 0) {
    uploadMsg.textContent = "Select a file or folder first.";
    return;
  }
  uploadMsg.textContent = `Uploading ${files.length} file(s)...`;
  let successCount = 0;
  const failures = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (file.name.startsWith('.')) continue; // skip hidden files like .DS_Store

    const formData = new FormData();
    formData.append("file", file);
    try {
      uploadMsg.textContent = `Uploading ${i + 1}/${files.length}: ${file.name}...`;
      const res = await fetch("/api/documents", { method: "POST", body: formData });
      if (res.ok) {
        successCount++;
      } else {
        const data = await res.json().catch(() => ({}));
        failures.push(`${file.name}: ${data.error || `Upload failed (${res.status})`}`);
      }
    } catch (err) {
      console.error("Upload failed for", file.name, err);
      failures.push(`${file.name}: ${err.message}`);
    }
  }
  uploadMsg.textContent = failures.length
    ? `Queued ${successCount} file(s), ${failures.length} failed.`
    : `Queued ${successCount} file(s).`;
  if (failures.length) {
    addMessage(`Upload issues:\n${failures.slice(0, 5).join("\n")}`, "assistant");
  }
  fileInput.value = "";
  const folderInput = document.getElementById("folderInput");
  if (folderInput) folderInput.value = "";
  await fetchDocs();
}

uploadBtn.addEventListener("click", () => uploadFiles(fileInput.files));

const folderInput = document.getElementById("folderInput");
const uploadFolderBtn = document.getElementById("uploadFolderBtn");

uploadFolderBtn.addEventListener("click", () => {
  folderInput.click();
});

folderInput.addEventListener("change", () => {
  if (folderInput.files.length > 0) {
    uploadFiles(folderInput.files);
  }
});

const driveUrlInput = document.getElementById("driveUrlInput");
const driveImportBtn = document.getElementById("driveImportBtn");

driveImportBtn.addEventListener("click", async () => {
  const url = driveUrlInput.value.trim();
  if (!url) {
    uploadMsg.textContent = "Paste a Drive URL first.";
    return;
  }
  uploadMsg.textContent = "Fetching from Drive...";
  try {
    const res = await fetch("/api/documents/drive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Import failed");
    uploadMsg.textContent = `Queued ${data.count} file(s) from Drive.`;
    driveUrlInput.value = "";
    await fetchDocs();
  } catch (err) {
    uploadMsg.textContent = `Drive import error: ${err.message}`;
    console.error(err);
  }
});

sendBtn.addEventListener("click", async () => {
  const message = chatInput.value.trim();
  if (!message) return;
  const scopedDocIds = Array.from(selectedDocIds);
  if (scopedDocIds.length === 0) {
    addMessage("Select at least one ready document before asking a question.", "assistant");
    return;
  }

  addMessage(message, "user");
  chatInput.value = "";
  const includeSummaries = !!includeDocSummaries?.checked;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        doc_ids: scopedDocIds,
        include_document_summaries: includeSummaries,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Chat failed (${res.status})`);
    }
    const responseIncludeSummaries = data.include_document_summaries !== false;
    addMessage(renderAssistantResponse(data.answer, data.sources || [], responseIncludeSummaries));
  } catch (error) {
    addMessage(`Chat failed: ${error.message}`, "assistant");
  }
});

selectAllDocs.addEventListener("change", () => {
  const readyDocIds = getReadyDocIds(docsCache);
  if (selectAllDocs.checked) {
    scopeMode = "all";
    selectedDocIds = new Set(readyDocIds);
  } else {
    scopeMode = "custom";
    selectedDocIds = new Set();
  }
  renderDocs(docsCache);
});

docList.addEventListener("change", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || !target.classList.contains("doc-select-checkbox")) {
    return;
  }
  const docId = target.dataset.docId;
  if (!docId) return;
  scopeMode = "custom";
  if (target.checked) {
    selectedDocIds.add(docId);
  } else {
    selectedDocIds.delete(docId);
  }
  updateScopeControls(docsCache);
});

fetchHealth();
fetchDocs();

docList.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement) || !target.classList.contains("delete-btn")) {
    return;
  }

  const docId = target.dataset.docId;
  if (!docId) return;

  const confirmed = window.confirm("Delete this document and all indexed chunks?");
  if (!confirmed) return;

  target.disabled = true;
  target.textContent = "Deleting...";
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      const message = data.error || `Delete failed (${res.status})`;
      throw new Error(message);
    }
    await fetchDocs();
    addMessage(`Deleted document ${docId}.`, "assistant");
  } catch (error) {
    addMessage(`Delete failed: ${error.message}`, "assistant");
    target.disabled = false;
    target.textContent = "Delete";
  }
});

deleteAllBtn.addEventListener("click", async () => {
  const confirmed = window.confirm("Delete ALL documents and indexed chunks?");
  if (!confirmed) return;

  deleteAllBtn.disabled = true;
  deleteAllBtn.textContent = "Deleting...";
  try {
    const res = await fetch("/api/documents", { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      const message = data.error || `Delete all failed (${res.status})`;
      throw new Error(message);
    }
    const data = await res.json().catch(() => ({}));
    await fetchDocs();
    addMessage(`Deleted ${data.deleted_documents ?? 0} documents.`, "assistant");
  } catch (error) {
    addMessage(`Delete all failed: ${error.message}`, "assistant");
  } finally {
    deleteAllBtn.disabled = false;
    deleteAllBtn.textContent = "Delete All";
  }
});
