const messagesEl = document.getElementById('messages');
const ragEl = document.getElementById('ragInspector');
const inputEl = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderInspector(chunks) {
  if (!chunks.length) {
    ragEl.innerHTML = '<p>No chunks retrieved yet.</p>';
    return;
  }

  ragEl.innerHTML = chunks.map((chunk) => `
    <div class="chunk-card">
      <div class="chunk-meta">Score: ${chunk.similarity_score.toFixed(3)} · Page: ${chunk.page_number} · Source: ${chunk.source_document}</div>
      <div class="chunk-text">${chunk.text}</div>
    </div>
  `).join('');
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData
  });
  const data = await response.json();
  appendMessage('assistant', `Uploaded ${data.document_name} with ${data.chunks_added} chunks.`);
}

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message) return;

  appendMessage('user', message);
  inputEl.value = '';

  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  const data = await response.json();
  appendMessage('assistant', data.answer);
  renderInspector(data.retrieved_chunks || []);
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    sendMessage();
  }
});
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) uploadFile(file);
});

document.addEventListener('paste', async (event) => {
  const clipboardItems = event.clipboardData?.items || [];
  const pdfItem = Array.from(clipboardItems).find((item) => item.type === 'application/pdf');
  if (!pdfItem) return;
  const file = pdfItem.getAsFile();
  if (file) {
    event.preventDefault();
    await uploadFile(file);
  }
});
