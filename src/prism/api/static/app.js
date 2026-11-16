const PrismApp = {
  _showTrace(traceEl, mode, data) {
    if (!traceEl || !data.trace_id) return;
    const q = mode === 'agent'
      ? `run_id=${encodeURIComponent(data.trace_id)}`
      : `trace_id=${encodeURIComponent(data.trace_id)}`;
    traceEl.classList.remove('hidden');
    traceEl.innerHTML = `trace: <a href="/trace?${q}">${data.trace_id}</a> · mode=${data.retrieval_mode || mode}`;
  },

  _showCitations(sourcesEl, citations) {
    sourcesEl.innerHTML = (citations || []).map((c) => `
      <div class="source">
        <div class="meta">[${c.n}] ${c.title} · score=${c.score}${c.section_path ? ' · ' + c.section_path : ''}</div>
      </div>`).join('');
  },

  async chat() {
    const query = document.getElementById('query').value.trim();
    const top_k = Number(document.getElementById('top_k').value || 5);
    const modeEl = document.getElementById('mode');
    const mode = modeEl ? modeEl.value : 'rag';
    const answerEl = document.getElementById('answer');
    const sourcesEl = document.getElementById('sources');
    const traceEl = document.getElementById('trace-link');
    answerEl.classList.remove('hidden');
    answerEl.textContent = mode === 'agent' ? 'Агент…' : 'Генерация…';
    sourcesEl.innerHTML = '';
    if (traceEl) { traceEl.classList.add('hidden'); traceEl.innerHTML = ''; }
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k, mode }),
    });
    const data = await res.json();
    if (!res.ok) {
      answerEl.textContent = data.detail || 'Ошибка';
      return;
    }
    answerEl.textContent = data.answer;
    this._showTrace(traceEl, mode, data);
    this._showCitations(sourcesEl, data.citations);
  },

  async chatStream() {
    const query = document.getElementById('query').value.trim();
    const top_k = Number(document.getElementById('top_k').value || 5);
    const mode = document.getElementById('mode')?.value || 'rag';
    const answerEl = document.getElementById('answer');
    const sourcesEl = document.getElementById('sources');
    const traceEl = document.getElementById('trace-link');
    answerEl.classList.remove('hidden');
    answerEl.textContent = '';
    sourcesEl.innerHTML = '';
    if (traceEl) { traceEl.classList.add('hidden'); traceEl.innerHTML = ''; }
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k, mode }),
    });
    if (!res.ok || !res.body) {
      answerEl.textContent = 'stream error';
      return;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let event = 'message';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop() || '';
      for (const block of parts) {
        let dataLine = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          if (line.startsWith('data:')) dataLine += line.slice(5).trim();
        }
        if (!dataLine) continue;
        let data;
        try { data = JSON.parse(dataLine); } catch { continue; }
        if (event === 'token') answerEl.textContent += (data.text || '');
        if (event === 'step') answerEl.textContent += `\n· ${data.kind || 'step'}`;
        if (event === 'final') {
          answerEl.textContent = data.answer || answerEl.textContent;
          this._showTrace(traceEl, mode, data);
          this._showCitations(sourcesEl, data.citations);
        }
        if (event === 'error') answerEl.textContent = data.detail || 'error';
      }
    }
  },

  async searchOnly() {
    const query = document.getElementById('query').value.trim();
    const top_k = Number(document.getElementById('top_k').value || 5);
    const answerEl = document.getElementById('answer');
    const sourcesEl = document.getElementById('sources');
    answerEl.classList.add('hidden');
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k }),
    });
    const data = await res.json();
    if (!res.ok) {
      sourcesEl.textContent = data.detail || 'Ошибка';
      return;
    }
    sourcesEl.innerHTML = (data.results || []).map((r, i) => `
      <div class="source">
        <div class="meta">[${i + 1}] ${r.document_name} · score=${r.score}</div>
        <div>${r.text.slice(0, 400)}${r.text.length > 400 ? '…' : ''}</div>
      </div>`).join('');
  },

  async uploadFile() {
    const file = document.getElementById('file').files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch('/api/documents/upload', { method: 'POST', body: fd });
    const data = await res.json();
    const el = document.getElementById('job-status');
    if (!res.ok) { el.textContent = data.detail || 'Ошибка'; return; }
    el.textContent = `job #${data.job_id} для document #${data.document_id}`;
    this.pollJob(data.job_id);
  },

  async uploadUrl() {
    const url = document.getElementById('url').value.trim();
    const res = await fetch('/api/documents/from-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    const el = document.getElementById('job-status');
    if (!res.ok) { el.textContent = data.detail || 'Ошибка'; return; }
    el.textContent = `job #${data.job_id} для document #${data.document_id}`;
    this.pollJob(data.job_id);
  },

  async pollJob(jobId) {
    const el = document.getElementById('job-status');
    const tick = async () => {
      const res = await fetch(`/api/jobs/${jobId}`);
      const j = await res.json();
      el.textContent = `job #${j.id}: ${j.status} ${j.progress}% (${j.current_step || ''})`;
      if (j.status === 'completed' || j.status === 'failed') {
        this.loadDocuments();
        return;
      }
      setTimeout(tick, 1000);
    };
    tick();
  },

  async loadDocuments() {
    const box = document.getElementById('docs');
    if (!box) return;
    const res = await fetch('/api/documents?page_size=50');
    const data = await res.json();
    box.innerHTML = (data.items || []).map((d) => `
      <div class="doc-row">
        <div>
          <strong>${d.name}</strong>
          <span class="badge ${d.status === 'indexed' ? 'ok' : d.status === 'error' ? 'err' : ''}">${d.status}</span>
          <div class="muted">${d.num_chunks} chunks · ${d.space}</div>
        </div>
        <button class="btn btn-ghost" onclick="PrismApp.deleteDoc(${d.id})">Удалить</button>
      </div>`).join('') || '<p class="muted">Пока пусто</p>';
  },

  async deleteDoc(id) {
    await fetch(`/api/documents/${id}`, { method: 'DELETE' });
    this.loadDocuments();
  },

  async saveSettings(form) {
    const fd = new FormData(form);
    const body = Object.fromEntries(fd.entries());
    body.chunk_size = Number(body.chunk_size);
    body.chunk_overlap = Number(body.chunk_overlap);
    body.top_k = Number(body.top_k);
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    document.getElementById('settings-msg').textContent = res.ok
      ? JSON.stringify(data, null, 2)
      : (data.detail || 'Ошибка');
  },
};
