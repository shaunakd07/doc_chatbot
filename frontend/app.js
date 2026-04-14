const statusEl = document.getElementById("status");
const docList = document.getElementById("docList");
const deleteAllBtn = document.getElementById("deleteAllBtn");
const uploadBtn = document.getElementById("uploadBtn");
const uploadMsg = document.getElementById("uploadMsg");
const fileInput = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const uploadFolderBtn = document.getElementById("uploadFolderBtn");
const driveUrlInput = document.getElementById("driveUrlInput");
const driveImportBtn = document.getElementById("driveImportBtn");
const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const includeDocSummaries = document.getElementById("includeDocSummaries");
const selectAllDocs = document.getElementById("selectAllDocs");
const scopeSummary = document.getElementById("scopeSummary");
const tabButtons = Array.from(document.querySelectorAll(".tab-btn[data-tab]"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel[data-tab-panel]"));
const eventLog = document.getElementById("eventLog");
const clearLogsBtn = document.getElementById("clearLogsBtn");
const TRASH_ICON_SRC = "/icons/trash-can-icon-vector-13490171.avif";
const DEFAULT_TAB = "chat";
const ACTIVITY_LOG_LIMIT = 400;
const SUPPRESSED_ACTIVITY_LOG_EVENTS = new Set(["ui.tab.changed"]);

let docsCache = [];
let selectedDocIds = new Set();
let scopeMode = "all";
let docsPollingTimer = null;
let docsSnapshotSignature = "";
let activityLogs = [];
let logSequence = 0;
let conversationId = null;

const UI_KEYS = {
  activeTab: "ui.active_tab",
};

function setActiveTab(tabName, options = {}) {
  const { persist = true, focusButton = false, logEvent = true } = options;
  const nextTab = tabButtons.some((btn) => btn.dataset.tab === tabName) ? tabName : DEFAULT_TAB;

  tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === nextTab;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
    button.setAttribute("tabindex", isActive ? "0" : "-1");
    if (isActive && focusButton) {
      button.focus();
    }
  });

  tabPanels.forEach((panel) => {
    const isActive = panel.dataset.tabPanel === nextTab;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });

  if (persist) {
    window.localStorage.setItem(UI_KEYS.activeTab, nextTab);
  }

  if (logEvent) {
    addActivityLog("ui.tab.changed", { tab: nextTab });
  }
}

function resolveInitialTab() {
  const validTabs = new Set(tabButtons.map((btn) => btn.dataset.tab));
  const tabFromQuery = new URL(window.location.href).searchParams.get("tab");
  if (tabFromQuery && validTabs.has(tabFromQuery)) {
    return tabFromQuery;
  }
  const storedTab = window.localStorage.getItem(UI_KEYS.activeTab);
  if (storedTab && validTabs.has(storedTab)) {
    return storedTab;
  }
  return DEFAULT_TAB;
}

function setupTabs() {
  if (!tabButtons.length || !tabPanels.length) return;

  tabButtons.forEach((button, index) => {
    button.addEventListener("click", () => {
      setActiveTab(button.dataset.tab || DEFAULT_TAB);
    });

    button.addEventListener("keydown", (event) => {
      if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) {
        return;
      }

      event.preventDefault();
      let targetIndex = index;
      if (event.key === "ArrowRight") {
        targetIndex = (index + 1) % tabButtons.length;
      }
      if (event.key === "ArrowLeft") {
        targetIndex = (index - 1 + tabButtons.length) % tabButtons.length;
      }
      if (event.key === "Home") {
        targetIndex = 0;
      }
      if (event.key === "End") {
        targetIndex = tabButtons.length - 1;
      }

      const target = tabButtons[targetIndex];
      if (!target) return;
      setActiveTab(target.dataset.tab || DEFAULT_TAB, { focusButton: true });
    });
  });

  const initialTab = resolveInitialTab();
  setActiveTab(initialTab, { persist: false, logEvent: false });
}

function normalizeLogLevel(level) {
  const normalized = String(level || "info").toLowerCase();
  if (normalized === "warn" || normalized === "error") {
    return normalized;
  }
  return "info";
}

function formatLogTimestamp(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  const ss = String(date.getSeconds()).padStart(2, "0");
  const ms = String(date.getMilliseconds()).padStart(3, "0");
  return `${y}-${m}-${d} ${hh}:${mm}:${ss}.${ms}`;
}

function normalizeLogDetail(detail) {
  if (detail == null) {
    return "";
  }
  if (typeof detail === "string") {
    return detail;
  }
  try {
    return JSON.stringify(detail, null, 2);
  } catch (_error) {
    return String(detail);
  }
}

function renderActivityLogs() {
  if (!eventLog) return;
  if (!activityLogs.length) {
    eventLog.innerHTML = '<div class="log-empty">No activity yet.</div>';
    return;
  }

  eventLog.innerHTML = activityLogs
    .map((entry) => {
      const level = normalizeLogLevel(entry.level);
      const detailText = normalizeLogDetail(entry.detail);
      return (
        `<article class="log-item log-level-${level}">` +
          `<div class="log-headline">` +
            `<span class="log-name">${escapeHtml(entry.name)}</span>` +
            `<span class="log-meta">` +
              `<span class="log-level-badge log-level-${level}">${escapeHtml(level)}</span>` +
              `<span>${escapeHtml(entry.timestamp)}</span>` +
            `</span>` +
          `</div>` +
          `<pre class="log-detail">${escapeHtml(detailText)}</pre>` +
        `</article>`
      );
    })
    .join("");
}

function addActivityLog(name, detail = {}, level = "info") {
  if (SUPPRESSED_ACTIVITY_LOG_EVENTS.has(String(name || ""))) {
    return;
  }
  const entry = {
    id: `log-${++logSequence}`,
    name: String(name || "event"),
    detail,
    level: normalizeLogLevel(level),
    timestamp: formatLogTimestamp(new Date()),
  };
  activityLogs.unshift(entry);
  if (activityLogs.length > ACTIVITY_LOG_LIMIT) {
    activityLogs = activityLogs.slice(0, ACTIVITY_LOG_LIMIT);
  }
  renderActivityLogs();
}

function setupLogInteractions() {
  renderActivityLogs();
  clearLogsBtn?.addEventListener("click", () => {
    const removedEntries = activityLogs.length;
    activityLogs = [];
    logSequence = 0;
    renderActivityLogs();
    addActivityLog("logs.cleared", { removed_entries: removedEntries });
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
  const metadata = getDocMetadata(doc);
  const reviewState = String(metadata.out_of_place_review_state || "").toLowerCase();
  if (reviewState === "needs_review") {
    const predictedTypeRaw = String(
      metadata.out_of_place_review_predicted_type || metadata.doc_type || "other"
    ).trim();
    const predictedType = predictedTypeRaw.replace(/_/g, " ");
    const confidenceValue = Number(metadata.out_of_place_review_confidence);
    const confidenceSuffix = Number.isFinite(confidenceValue) && confidenceValue > 0
      ? `, ${confidenceValue.toFixed(2)}`
      : "";
    return `${doc.status}${ready ? "" : " (not selectable yet)"} | please review (${predictedType}${confidenceSuffix})`;
  }
  return `${doc.status}${ready ? "" : " (not selectable yet)"}`;
}

function shouldPollDocs(docs) {
  return (docs || []).some((doc) => {
    const status = String(doc.status || "").toLowerCase();
    if (status === "queued" || status === "processing") {
      return true;
    }
    const metadata = getDocMetadata(doc);
    const hasReviewContext = Boolean(metadata.folder_id) && Boolean(metadata.expected_doc_type);
    const hasReviewResult = Boolean(metadata.out_of_place_review_last_checked_at)
      || Boolean(metadata.out_of_place_review_state);
    return hasReviewContext && !hasReviewResult;
  });
}

function buildDocsSnapshot(docs) {
  return (docs || [])
    .map((doc) => `${doc.id}:${doc.status}`)
    .sort()
    .join("|");
}

function summarizeDocs(docs) {
  const summary = {
    total: docs.length,
    ready: 0,
    queued: 0,
    processing: 0,
    failed: 0,
  };

  docs.forEach((doc) => {
    const status = String(doc.status || "").toLowerCase();
    if (status === "ready") summary.ready += 1;
    if (status === "queued") summary.queued += 1;
    if (status === "processing") summary.processing += 1;
    if (status === "failed") summary.failed += 1;
  });

  return summary;
}

function syncDocsPolling() {
  const needsPolling = shouldPollDocs(docsCache);
  if (needsPolling && !docsPollingTimer) {
    addActivityLog("documents.polling.started", { interval_ms: 1500 });
    docsPollingTimer = setInterval(() => {
      fetchDocs("poll").catch((err) => {
        console.error("Doc polling failed", err);
        addActivityLog("documents.polling.error", { error: err.message }, "error");
      });
    }, 1500);
    return;
  }
  if (!needsPolling && docsPollingTimer) {
    clearInterval(docsPollingTimer);
    docsPollingTimer = null;
    addActivityLog("documents.polling.stopped", {});
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
  if (!scopeSummary || !selectAllDocs) return;
  const readyDocIds = getReadyDocIds(docs);
  const selectedCount = readyDocIds.filter((docId) => selectedDocIds.has(docId)).length;
  scopeSummary.textContent = `${selectedCount}/${readyDocIds.length} selected`;

  selectAllDocs.disabled = readyDocIds.length === 0;
  selectAllDocs.indeterminate = selectedCount > 0 && selectedCount < readyDocIds.length;
  selectAllDocs.checked = readyDocIds.length > 0 && selectedCount === readyDocIds.length;
}

async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (statusEl) {
      statusEl.textContent = data.vlm_enabled ? "VLM enabled" : "VLM disabled";
    }
    addActivityLog("health.checked", { vlm_enabled: !!data.vlm_enabled });
  } catch (error) {
    if (statusEl) {
      statusEl.textContent = "Health check failed";
    }
    addActivityLog("health.error", { error: error.message }, "error");
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
    const metadata = getDocMetadata(doc);
    const isReviewWarning = String(metadata.out_of_place_review_state || "").toLowerCase() === "needs_review";
    const li = document.createElement("li");
    li.className = isReviewWarning ? "doc-item doc-item-review-warning" : "doc-item";
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

async function fetchDocs(reason = "manual") {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    docsCache = data.documents || [];
    syncSelectionToDocs(docsCache);
    renderDocs(docsCache);
    const nextSnapshot = buildDocsSnapshot(docsCache);
    if (nextSnapshot !== docsSnapshotSignature) {
      docsSnapshotSignature = nextSnapshot;
      addActivityLog("documents.synced", {
        reason,
        ...summarizeDocs(docsCache),
      });
    }
    syncDocsPolling();
  } catch (error) {
    addActivityLog("documents.sync.error", { reason, error: error.message }, "error");
    throw error;
  }
}

function addMessage(text, role = "assistant") {
  if (!chatLog) return;
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `<div class="meta">${role}</div><div>${text}</div>`;
  chatLog.prepend(div);
  chatLog.scrollTop = 0;
}

function addThinkingIndicator() {
  if (!chatLog) return null;
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
  chatLog.prepend(div);
  chatLog.scrollTop = 0;
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

async function parseApiJson(res) {
  const raw = await res.text();
  if (!raw) {
    return {};
  }
  try {
    return JSON.parse(raw);
  } catch (_error) {
    return { error: raw.trim() || `Request failed (${res.status})` };
  }
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
    if (uploadMsg) {
      uploadMsg.textContent = "Select a file or folder first.";
    }
    addActivityLog("upload.blocked", { reason: "no_files_selected" }, "warn");
    return;
  }

  if (uploadMsg) {
    uploadMsg.textContent = `Uploading ${files.length} file(s)...`;
  }
  addActivityLog("upload.started", {
    file_count: files.length,
    files: Array.from(files).slice(0, 40).map((file) => file.name),
  });

  let successCount = 0;
  const failures = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (file.name.startsWith(".")) {
      addActivityLog("upload.skipped_hidden", { file: file.name }, "warn");
      continue;
    }

    const formData = new FormData();
    formData.append("file", file);
    try {
      if (uploadMsg) {
        uploadMsg.textContent = `Uploading ${i + 1}/${files.length}: ${file.name}...`;
      }
      const res = await fetch("/api/documents", { method: "POST", body: formData });
      if (res.ok) {
        successCount++;
        addActivityLog("upload.file.queued", { file: file.name, index: i + 1, total: files.length });
      } else {
        const data = await res.json().catch(() => ({}));
        const message = data.error || `Upload failed (${res.status})`;
        failures.push(`${file.name}: ${message}`);
        addActivityLog("upload.file.failed", { file: file.name, status: res.status, error: message }, "error");
      }
    } catch (err) {
      console.error("Upload failed for", file.name, err);
      failures.push(`${file.name}: ${err.message}`);
      addActivityLog("upload.file.error", { file: file.name, error: err.message }, "error");
    }
  }

  if (uploadMsg) {
    uploadMsg.textContent = failures.length
      ? `Queued ${successCount} file(s), ${failures.length} failed.`
      : `Queued ${successCount} file(s).`;
  }
  addActivityLog("upload.completed", {
    queued_files: successCount,
    failed_files: failures.length,
    failures: failures.slice(0, 20),
  }, failures.length ? "warn" : "info");

  if (failures.length) {
    addMessage(`Upload issues:\n${failures.slice(0, 5).join("\n")}`, "assistant");
  }
  if (fileInput) {
    fileInput.value = "";
  }
  if (folderInput) {
    folderInput.value = "";
  }
  await fetchDocs("upload");
}

uploadBtn?.addEventListener("click", () => uploadFiles(fileInput?.files));

uploadFolderBtn?.addEventListener("click", () => {
  folderInput?.click();
});

folderInput?.addEventListener("change", () => {
  if (folderInput.files.length > 0) {
    addActivityLog("upload.folder.selected", { file_count: folderInput.files.length });
    uploadFiles(folderInput.files);
  }
});

driveImportBtn?.addEventListener("click", async () => {
  const url = driveUrlInput?.value.trim() || "";
  if (!url) {
    if (uploadMsg) {
      uploadMsg.textContent = "Paste a Drive URL first.";
    }
    addActivityLog("drive_import.blocked", { reason: "missing_url" }, "warn");
    return;
  }
  if (uploadMsg) {
    uploadMsg.textContent = "Fetching from Drive...";
  }
  addActivityLog("drive_import.started", { url });
  try {
    const res = await fetch("/api/documents/drive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Import failed");
    if (uploadMsg) {
      uploadMsg.textContent = `Queued ${data.count} file(s) from Drive.`;
    }
    if (driveUrlInput) {
      driveUrlInput.value = "";
    }
    addActivityLog("drive_import.completed", { queued_files: data.count });
    await fetchDocs("drive_import");
  } catch (err) {
    if (uploadMsg) {
      uploadMsg.textContent = `Drive import error: ${err.message}`;
    }
    addActivityLog("drive_import.error", { error: err.message, url }, "error");
    console.error(err);
  }
});

sendBtn?.addEventListener("click", async () => {
  const message = chatInput?.value.trim() || "";
  if (!message) return;
  const scopedDocIds = Array.from(selectedDocIds);
  if (scopedDocIds.length === 0) {
    addMessage("Select at least one ready document before asking a question.", "assistant");
    addActivityLog("chat.blocked", { reason: "no_ready_documents_selected" }, "warn");
    return;
  }

  addMessage(message, "user");
  if (chatInput) {
    chatInput.value = "";
  }
  const includeSummaries = !!includeDocSummaries?.checked;
  const startedAt = Date.now();
  addActivityLog("chat.request.sent", {
    message,
    doc_ids: scopedDocIds,
    include_document_summaries: includeSummaries,
    conversation_id: conversationId,
  });

  const thinkingIndicator = addThinkingIndicator();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        doc_ids: scopedDocIds,
        include_document_summaries: includeSummaries,
        conversation_id: conversationId,
      }),
    });
    const data = await parseApiJson(res);
    if (!res.ok) {
      throw new Error(data.error || data.detail || `Chat failed (${res.status})`);
    }
    const nextConversationId = String(data.conversation_id || "").trim();
    if (nextConversationId) {
      conversationId = nextConversationId;
    }
    const responseIncludeSummaries = data.include_document_summaries !== false;
    removeThinkingIndicator(thinkingIndicator);
    addMessage(renderAssistantResponse(data.answer, data.sources || [], responseIncludeSummaries));
    addActivityLog("chat.response.received", {
      duration_ms: Date.now() - startedAt,
      source_count: Array.isArray(data.sources) ? data.sources.length : 0,
      include_document_summaries: responseIncludeSummaries,
      conversation_id: conversationId,
      answer_preview: String(data.answer || "").slice(0, 220),
    });
  } catch (error) {
    removeThinkingIndicator(thinkingIndicator);
    addMessage(`Chat failed: ${error.message}`, "assistant");
    addActivityLog("chat.response.error", { duration_ms: Date.now() - startedAt, error: error.message }, "error");
  }
});

chatInput?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  sendBtn?.click();
});

selectAllDocs?.addEventListener("change", () => {
  const readyDocIds = getReadyDocIds(docsCache);
  if (selectAllDocs.checked) {
    scopeMode = "all";
    selectedDocIds = new Set(readyDocIds);
  } else {
    scopeMode = "custom";
    selectedDocIds = new Set();
  }
  renderDocs(docsCache);
  addActivityLog("documents.scope.select_all", {
    checked: selectAllDocs.checked,
    selected_count: selectedDocIds.size,
    ready_count: readyDocIds.length,
  });
});

docList?.addEventListener("change", (event) => {
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
  addActivityLog("documents.scope.toggled", {
    doc_id: docId,
    selected: target.checked,
    selected_count: selectedDocIds.size,
  });
});

docList?.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof Element)) {
    return;
  }
  const deleteBtn = target.closest("button.delete-btn");
  if (!(deleteBtn instanceof HTMLButtonElement)) return;

  const docId = deleteBtn.dataset.docId;
  if (!docId) return;

  const confirmed = window.confirm("Delete this document and all indexed chunks?");
  if (!confirmed) {
    addActivityLog("documents.delete.cancelled", { doc_id: docId }, "warn");
    return;
  }

  deleteBtn.disabled = true;
  deleteBtn.classList.add("is-loading");
  addActivityLog("documents.delete.started", { doc_id: docId });

  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      const message = data.error || `Delete failed (${res.status})`;
      throw new Error(message);
    }
    await fetchDocs("delete_single");
    addMessage(`Deleted document ${docId}.`, "assistant");
    addActivityLog("documents.delete.completed", { doc_id: docId });
  } catch (error) {
    addMessage(`Delete failed: ${error.message}`, "assistant");
    deleteBtn.disabled = false;
    deleteBtn.classList.remove("is-loading");
    addActivityLog("documents.delete.error", { doc_id: docId, error: error.message }, "error");
  }
});

deleteAllBtn?.addEventListener("click", async () => {
  const confirmed = window.confirm("Delete ALL documents and indexed chunks?");
  if (!confirmed) {
    addActivityLog("documents.delete_all.cancelled", {}, "warn");
    return;
  }

  deleteAllBtn.disabled = true;
  deleteAllBtn.textContent = "Deleting...";
  addActivityLog("documents.delete_all.started", {});
  try {
    const res = await fetch("/api/documents", { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      const message = data.error || `Delete all failed (${res.status})`;
      throw new Error(message);
    }
    const data = await res.json().catch(() => ({}));
    await fetchDocs("delete_all");
    addMessage(`Deleted ${data.deleted_documents ?? 0} documents.`, "assistant");
    addActivityLog("documents.delete_all.completed", {
      deleted_documents: data.deleted_documents ?? 0,
    });
  } catch (error) {
    addMessage(`Delete all failed: ${error.message}`, "assistant");
    addActivityLog("documents.delete_all.error", { error: error.message }, "error");
  } finally {
    deleteAllBtn.disabled = false;
    deleteAllBtn.textContent = "Delete All";
  }
});

setupTabs();
setupLogInteractions();
addActivityLog("ui.ready", {
  tabs: tabButtons.map((btn) => btn.dataset.tab),
  default_tab: window.localStorage.getItem(UI_KEYS.activeTab) || DEFAULT_TAB,
});

fetchHealth();
fetchDocs("initial").catch((error) => {
  console.error(error);
  addActivityLog("documents.initial_load.error", { error: error.message }, "error");
});
