const statusEl = document.getElementById("status");
const docList = document.getElementById("docList");
const deleteAllBtn = document.getElementById("deleteAllBtn");
const uploadBtn = document.getElementById("uploadBtn");
const uploadMsg = document.getElementById("uploadMsg");
const fileInput = document.getElementById("fileInput");
const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

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
    return;
  }
  docs.forEach((doc) => {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `
      <div class="doc-main">
        <strong>${doc.filename}</strong>
        <span>${doc.status}</span>
      </div>
      <button class="delete-btn" data-doc-id="${doc.id}" type="button">Delete</button>
    `;
    docList.appendChild(li);
  });
}

async function fetchDocs() {
  const res = await fetch("/api/documents");
  const data = await res.json();
  renderDocs(data.documents || []);
}

function addMessage(text, role = "assistant") {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `<div class="meta">${role}</div><div>${text}</div>`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function uploadFiles(files) {
  if (!files || files.length === 0) {
    uploadMsg.textContent = "Select a file or folder first.";
    return;
  }
  uploadMsg.textContent = `Uploading ${files.length} file(s)...`;
  let successCount = 0;
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (file.name.startsWith('.')) continue; // skip hidden files like .DS_Store

    const formData = new FormData();
    formData.append("file", file);
    try {
      uploadMsg.textContent = `Uploading ${i + 1}/${files.length}: ${file.name}...`;
      const res = await fetch("/api/documents", { method: "POST", body: formData });
      if (res.ok) successCount++;
    } catch (err) {
      console.error("Upload failed for", file.name, err);
    }
  }
  uploadMsg.textContent = `Queued ${successCount} file(s).`;
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
  addMessage(message, "user");
  chatInput.value = "";
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const data = await res.json();
  const sources = (data.sources || [])
    .map((s) => {
      const name = s.doc_filename || "unknown";
      return `[${name} | ${s.doc_id}:${s.page}] ${s.content.substring(0, 140)}...`;
    })
    .join("<br />");
  addMessage(`${data.answer}<br /><br /><strong>Sources</strong><br />${sources}`);
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
