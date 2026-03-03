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
const workspace = document.getElementById("workspace");
const docsSidebar = document.getElementById("docsSidebar");
const mainArea = document.getElementById("mainArea");
const chatPane = document.getElementById("chatPane");
const sidebarResizer = document.getElementById("sidebarResizer");
const chatResizer = document.getElementById("chatResizer");
const collapseSidebarBtn = document.getElementById("collapseSidebarBtn");
const expandSidebarBtn = document.getElementById("expandSidebarBtn");
const TRASH_ICON_SRC = "/icons/trash-can-icon-vector-13490171.avif";

let docsCache = [];
let selectedDocIds = new Set();
let scopeMode = "all";
let docsPollingTimer = null;

const SIDEBAR_MIN = 240;
const SIDEBAR_MAX = 640;
const CHAT_MIN = 460;
const UI_KEYS = {
  sidebarWidth: "ui.sidebar.width",
  sidebarCollapsed: "ui.sidebar.collapsed",
  layoutVersion: "ui.layout.version",
};
const UI_LAYOUT_VERSION = "2";

function getChatMinWidth(mainWidth) {
  return Math.max(CHAT_MIN, Math.round(mainWidth * 0.6));
}

function clamp(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

function getStoredNumber(key) {
  const value = Number(window.localStorage.getItem(key));
  return Number.isFinite(value) ? value : null;
}

function setSidebarCollapsed(collapsed) {
  if (!workspace) return;
  workspace.classList.toggle("sidebar-collapsed", !!collapsed);
  if (collapseSidebarBtn) {
    collapseSidebarBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    collapseSidebarBtn.textContent = collapsed ? "Expand" : "Collapse";
  }
  if (expandSidebarBtn) {
    expandSidebarBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  }
  window.localStorage.setItem(UI_KEYS.sidebarCollapsed, collapsed ? "1" : "0");
}

function applyInitialLayoutPrefs() {
  if (!docsSidebar || !chatPane) return;

  const savedLayoutVersion = window.localStorage.getItem(UI_KEYS.layoutVersion);
  if (savedLayoutVersion !== UI_LAYOUT_VERSION) {
    window.localStorage.setItem(UI_KEYS.layoutVersion, UI_LAYOUT_VERSION);
  }

  const sidebarWidthStored = getStoredNumber(UI_KEYS.sidebarWidth);
  const sidebarWidth = clamp(sidebarWidthStored ?? 320, SIDEBAR_MIN, SIDEBAR_MAX);
  docsSidebar.style.setProperty("--sidebar-width", `${sidebarWidth}px`);
  docsSidebar.style.width = `${sidebarWidth}px`;

  const initiallyCollapsed = window.localStorage.getItem(UI_KEYS.sidebarCollapsed) === "1";
  setSidebarCollapsed(initiallyCollapsed);

  chatPane.style.setProperty("--chat-width", "100%");
  chatPane.style.width = "100%";
}

function installHorizontalResizer(handle, onMove, onEnd) {
  if (!handle) return;
  let dragState = null;

  const stop = () => {
    if (!dragState) return;
    dragState = null;
    document.body.classList.remove("resizing");
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", stop);
    if (typeof onEnd === "function") onEnd();
  };

  const move = (event) => {
    if (!dragState) return;
    onMove(event, dragState);
  };

  handle.addEventListener("mousedown", (event) => {
    event.preventDefault();
    dragState = {
      startX: event.clientX,
      sidebarWidth: docsSidebar ? docsSidebar.getBoundingClientRect().width : 0,
      chatWidth: chatPane ? chatPane.getBoundingClientRect().width : 0,
      mainWidth: mainArea ? mainArea.getBoundingClientRect().width : 0,
    };
    document.body.classList.add("resizing");
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", stop);
  });
}

function setupLayoutInteractions() {
  applyInitialLayoutPrefs();

  if (collapseSidebarBtn) {
    collapseSidebarBtn.addEventListener("click", () => {
      const collapsed = workspace?.classList.contains("sidebar-collapsed");
      setSidebarCollapsed(!collapsed);
    });
  }

  if (expandSidebarBtn) {
    expandSidebarBtn.addEventListener("click", () => setSidebarCollapsed(false));
  }

  installHorizontalResizer(
    sidebarResizer,
    (event, state) => {
      if (!docsSidebar || !workspace || workspace.classList.contains("sidebar-collapsed")) return;
      const next = clamp(state.sidebarWidth + (event.clientX - state.startX), SIDEBAR_MIN, SIDEBAR_MAX);
      docsSidebar.style.setProperty("--sidebar-width", `${next}px`);
      docsSidebar.style.width = `${next}px`;
    },
    () => {
      if (!docsSidebar) return;
      window.localStorage.setItem(UI_KEYS.sidebarWidth, `${Math.round(docsSidebar.getBoundingClientRect().width)}`);
    }
  );

  installHorizontalResizer(
    chatResizer,
    (event, state) => {
      if (!chatPane || !mainArea) return;
      const minWidth = getChatMinWidth(state.mainWidth);
      const maxWidth = Math.max(minWidth, state.mainWidth - 20);
      const next = clamp(state.chatWidth + (event.clientX - state.startX), minWidth, maxWidth);
      chatPane.style.setProperty("--chat-width", `${next}px`);
      chatPane.style.width = `${next}px`;
    },
    () => {
      if (!chatPane || !mainArea) return;
      const width = Math.round(chatPane.getBoundingClientRect().width);
      const mainWidth = mainArea.getBoundingClientRect().width;
      const minWidth = getChatMinWidth(mainWidth);
      const maxWidth = Math.round(Math.max(minWidth, mainWidth - 20));
      if (width >= maxWidth - 2) {
        chatPane.style.setProperty("--chat-width", "100%");
        chatPane.style.width = "100%";
      }
    }
  );

  if (chatResizer) {
    chatResizer.addEventListener("dblclick", () => {
      if (!chatPane) return;
      chatPane.style.setProperty("--chat-width", "100%");
      chatPane.style.width = "100%";
    });
  }

  window.addEventListener("resize", () => {
    if (!chatPane || !mainArea) return;
    const explicitWidth = chatPane.style.width;
    if (!explicitWidth || explicitWidth === "100%") {
      chatPane.style.setProperty("--chat-width", "100%");
      chatPane.style.width = "100%";
      return;
    }
    const currentWidth = Math.round(chatPane.getBoundingClientRect().width);
    const mainWidth = mainArea.getBoundingClientRect().width;
    const minWidth = getChatMinWidth(mainWidth);
    const maxWidth = Math.max(minWidth, mainWidth - 20);
    const next = clamp(currentWidth, minWidth, maxWidth);
    if (next >= maxWidth - 2) {
      chatPane.style.setProperty("--chat-width", "100%");
      chatPane.style.width = "100%";
      return;
    }
    chatPane.style.setProperty("--chat-width", `${next}px`);
    chatPane.style.width = `${next}px`;
  });
}

function isDocReady(doc) {
  return String(doc.status || "").toLowerCase() === "ready";
}

function getDocMetadata(doc) {
  const meta = doc && typeof doc.metadata === "object" ? doc.metadata : {};
  return meta && !Array.isArray(meta) ? meta : {};
}

function clampPercent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.round(n)));
}

function getDocProgress(doc) {
  const metadata = getDocMetadata(doc);
  if (metadata.ingest_progress != null) {
    return clampPercent(metadata.ingest_progress);
  }
  const status = String(doc.status || "").toLowerCase();
  if (status === "ready") return 100;
  if (status === "processing") return 35;
  return 0;
}

function getDocStatusLabel(doc) {
  const ready = isDocReady(doc);
  return `${doc.status}${ready ? "" : " (not selectable yet)"}`;
}

function shouldPollDocs(docs) {
  return (docs || []).some((doc) => {
    const status = String(doc.status || "").toLowerCase();
    return status === "queued" || status === "processing";
  });
}

function syncDocsPolling() {
  const needsPolling = shouldPollDocs(docsCache);
  if (needsPolling && !docsPollingTimer) {
    docsPollingTimer = setInterval(() => {
      fetchDocs().catch((err) => console.error("Doc polling failed", err));
    }, 1500);
    return;
  }
  if (!needsPolling && docsPollingTimer) {
    clearInterval(docsPollingTimer);
    docsPollingTimer = null;
  }
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
  if (statusEl) {
    statusEl.textContent = data.vlm_enabled ? "VLM enabled" : "VLM disabled";
  }
}

function renderDocs(docs) {
  docList.innerHTML = "";
  if (!docs.length) {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.textContent = "No documents yet.";
    docList.appendChild(li);
    updateScopeControls(docs);
    return;
  }
  docs.forEach((doc) => {
    const ready = isDocReady(doc);
    const checked = ready && selectedDocIds.has(doc.id);
    const progressPct = getDocProgress(doc);
    const statusLower = String(doc.status || "").toLowerCase();
    const statusLabel = getDocStatusLabel(doc);
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `
      <div class="doc-main-wrap">
        <div class="doc-controls">
          <label class="doc-scope-toggle" title="${ready ? "Include this document in chat context" : "Only ready documents can be selected"}">
            <input class="doc-select-checkbox" data-doc-id="${doc.id}" type="checkbox" ${checked ? "checked" : ""} ${ready ? "" : "disabled"} />
          </label>
          <button class="delete-btn" data-doc-id="${doc.id}" type="button" aria-label="Delete document ${escapeHtml(doc.filename)}" title="Delete document">
            <img src="${TRASH_ICON_SRC}" alt="" />
          </button>
        </div>
        <div class="doc-main">
          <strong>${doc.filename}</strong>
          <span>${statusLabel}</span>
          <div class="doc-progress-wrap">
            <div class="doc-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progressPct}">
              <div class="doc-progress-fill ${statusLower}" style="width: ${progressPct}%"></div>
            </div>
            <span class="doc-progress-percent">${progressPct}%</span>
          </div>
        </div>
      </div>
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
  syncDocsPolling();
}

function addMessage(text, role = "assistant") {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `<div class="meta">${role}</div><div>${text}</div>`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addThinkingIndicator() {
  const div = document.createElement("div");
  div.className = "message assistant thinking-message";

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = "assistant";

  const indicator = document.createElement("div");
  indicator.className = "thinking-indicator";
  indicator.setAttribute("aria-label", "Thinking...");
  indicator.setAttribute("role", "status");

  const text = "Thinking...";
  [...text].forEach((char, index) => {
    const span = document.createElement("span");
    span.className = "thinking-letter";
    span.style.setProperty("--thinking-index", String(index));
    span.textContent = char;
    indicator.appendChild(span);
  });

  div.appendChild(meta);
  div.appendChild(indicator);
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function removeThinkingIndicator(indicatorEl) {
  if (!indicatorEl || !indicatorEl.parentNode) return;
  indicatorEl.parentNode.removeChild(indicatorEl);
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
  const thinkingIndicator = addThinkingIndicator();
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
    removeThinkingIndicator(thinkingIndicator);
    addMessage(renderAssistantResponse(data.answer, data.sources || [], responseIncludeSummaries));
  } catch (error) {
    removeThinkingIndicator(thinkingIndicator);
    addMessage(`Chat failed: ${error.message}`, "assistant");
  }
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  sendBtn.click();
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
setupLayoutInteractions();

docList.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof Element)) {
    return;
  }
  const deleteBtn = target.closest("button.delete-btn");
  if (!(deleteBtn instanceof HTMLButtonElement)) return;

  const docId = deleteBtn.dataset.docId;
  if (!docId) return;

  const confirmed = window.confirm("Delete this document and all indexed chunks?");
  if (!confirmed) return;

  deleteBtn.disabled = true;
  deleteBtn.classList.add("is-loading");
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
    deleteBtn.disabled = false;
    deleteBtn.classList.remove("is-loading");
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
