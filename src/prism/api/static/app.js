const PrismApp = {
  _lastCitations: [],

  _escapeAttr(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  },

  _escape(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  },

  _modeRu(mode) {
    const m = String(mode || '');
    const parts = m.split('+').filter(Boolean);
    const map = {
      dense: 'векторный поиск',
      lexical: 'лексический (BM25)',
      hybrid: 'гибридный поиск',
      space_weights: 'учёт весов пространств',
      rag: 'быстрый RAG',
      agent: 'агент',
    };
    const out = [];
    for (const p of parts) {
      if (p.startsWith('space:')) {
        const sp = p.slice(6);
        const spRu = { personal: 'личное', project: 'проект', shared: 'общее', all: 'все' }[sp] || sp;
        out.push(`фильтр: ${spRu}`);
      } else if (p.startsWith('rewrite:')) {
        out.push(`переписывание запроса (${p.slice(8)})`);
      } else if (p.startsWith('rerank:')) {
        out.push(`реранжирование (${p.slice(7)})`);
      } else {
        out.push(map[p] || p);
      }
    }
    return out.join(' · ') || m || '—';
  },

  _complexityRu(c) {
    return ({
      simple: 'простой',
      multi: 'многошаговый',
      verify: 'с проверкой',
      research: 'исследование',
    })[c] || c || '—';
  },

  _stepTitle(step) {
    const kind = step.kind || step.type || '';
    const tool = step.input?.tool || step.tool || step.output?.tool;
    const titles = {
      plan: 'План (оценка сложности запроса)',
      tool: tool ? `Инструмент: ${tool}` : 'Инструмент',
      delegation: 'Делегирование субагентам',
      synthesize: 'Сборка ответа',
      status: 'Статус',
      forced_synthesis: 'Принудительная сборка (лимит шагов)',
    };
    return titles[kind] || kind || 'Шаг';
  },

  _stepDetail(step) {
    const kind = step.kind || '';
    if (kind === 'plan') {
      const p = step.output || {};
      const roles = (p.roles || []).join(', ') || '—';
      const complexity = this._complexityRu(p.complexity);
      const steps = (p.steps || []).join(' → ') || '—';
      return `сложность: ${complexity}; роли: ${roles}; шаги: ${steps}`;
    }
    if (kind === 'tool') {
      const tool = step.input?.tool || '';
      const out = step.output;
      if (tool === 'search_kb' && out && typeof out === 'object') {
        const n = out.n ?? out.hits?.length ?? out.results?.length;
        return n != null ? `найдено фрагментов: ${n}` : 'поиск по базе';
      }
      if (tool === 'verify_citations') return 'проверка цитат и опоры на контекст';
      if (tool === 'web_search') return 'поиск в сети (или mock)';
      if (tool === 'build_comparison_table') return 'сводная таблица';
      if (tool === 'distill_run') return 'запись выводов в память';
      return tool || 'вызов инструмента';
    }
    if (kind === 'delegation') {
      const roles = (step.input?.roles || []).join(', ');
      return roles ? `роли: ${roles}` : 'запуск субагентов';
    }
    if (kind === 'synthesize') {
      return step.input?.revision ? 'повторная сборка после критика' : 'финальная формулировка';
    }
    if (typeof step.output === 'string') return step.output.slice(0, 160);
    return '';
  },

  _els() {
    return {
      panel: document.getElementById('run-panel'),
      status: document.getElementById('run-status'),
      statusText: document.querySelector('#run-status .run-status-text'),
      steps: document.getElementById('run-steps'),
      answer: document.getElementById('answer'),
      sources: document.getElementById('sources'),
      trace: document.getElementById('trace-link'),
      ask: document.getElementById('btn-ask'),
      search: document.getElementById('btn-search'),
    };
  },

  _setBusy(busy) {
    const { ask, search } = this._els();
    if (ask) ask.disabled = busy;
    if (search) search.disabled = busy;
  },

  _beginRun(label) {
    const { panel, status, statusText, steps, answer, sources, trace } = this._els();
    if (!panel) return;
    panel.classList.remove('hidden');
    panel.classList.add('running');
    if (status) status.classList.remove('done', 'error');
    if (statusText) statusText.textContent = label;
    if (steps) steps.innerHTML = '';
    if (answer) {
      answer.classList.add('hidden');
      answer.textContent = '';
    }
    if (sources) sources.innerHTML = '';
    if (trace) {
      trace.classList.add('hidden');
      trace.innerHTML = '';
    }
    this._setBusy(true);
  },

  _setStatus(text, { done = false, error = false } = {}) {
    const { panel, status, statusText } = this._els();
    if (statusText) statusText.textContent = text;
    if (status) {
      status.classList.toggle('done', done);
      status.classList.toggle('error', error);
    }
    if (panel && (done || error)) panel.classList.remove('running');
    if (done || error) this._setBusy(false);
  },

  _addStep(title, detail = '', { active = false } = {}) {
    const { steps } = this._els();
    if (!steps) return null;
    steps.querySelectorAll('.run-step.active').forEach((el) => el.classList.remove('active'));
    const li = document.createElement('li');
    li.className = 'run-step' + (active ? ' active' : '');
    li.innerHTML = `<div class="run-step-title">${this._escape(title)}</div>` +
      (detail ? `<div class="run-step-detail">${this._escape(detail)}</div>` : '');
    steps.appendChild(li);
    li.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    return li;
  },

  _renderAnswerHtml(text) {
    const escaped = this._escape(text || '');
    return escaped.replace(/\[(\d+)\]/g, (_, n) =>
      `<a href="#cite-${n}" class="cite-ref" data-cite="${n}" title="Открыть источник [${n}]">[${n}]</a>`
    );
  },

  _bindAnswerCitationClicks(answerEl) {
    if (!answerEl) return;
    answerEl.querySelectorAll('[data-cite]').forEach((el) => {
      el.addEventListener('click', (e) => {
        e.preventDefault();
        this.openCitation(Number(el.getAttribute('data-cite')));
      });
    });
  },

  _showAnswer(text) {
    const { answer } = this._els();
    if (!answer) return;
    answer.classList.remove('hidden');
    answer.innerHTML = this._renderAnswerHtml(text);
    this._bindAnswerCitationClicks(answer);
  },

  _appendAnswer(text) {
    const { answer } = this._els();
    if (!answer) return;
    answer.classList.remove('hidden');
    // During token stream keep plain text; final event re-renders with links.
    answer.textContent += text || '';
  },

  openCitation(n) {
    const c = (this._lastCitations || []).find((x) => Number(x.n) === Number(n));
    const drawer = document.getElementById('cite-drawer');
    const title = document.getElementById('cite-drawer-title');
    const meta = document.getElementById('cite-drawer-meta');
    const body = document.getElementById('cite-drawer-body');
    if (!drawer || !title || !meta || !body) return;
    if (!c) {
      title.textContent = `Источник [${n}]`;
      meta.textContent = 'Фрагмент не найден в ответе';
      body.textContent = '';
      drawer.classList.remove('hidden');
      return;
    }
    const kind = c.kind === 'web' || c.url ? 'web' : 'kb';
    title.textContent = `[${c.n}] ${c.title || (kind === 'web' ? 'веб' : 'документ')}`;
    const bits = [];
    if (kind === 'web') bits.push('веб');
    if (c.page != null) bits.push(`стр. ${c.page}`);
    if (c.doc_id) bits.push(`документ #${c.doc_id}`);
    if (c.chunk_id) bits.push(`фрагмент #${c.chunk_id}`);
    if (c.score != null && kind === 'kb') bits.push(`релевантность ${c.score}`);
    if (c.url) {
      meta.innerHTML =
        (bits.length ? this._escape(bits.join(' · ')) + ' · ' : '') +
        `<a href="${this._escapeAttr(c.url)}" target="_blank" rel="noopener noreferrer">${this._escape(c.url)}</a>`;
    } else {
      if (c.section_path) bits.push(c.section_path);
      meta.textContent = bits.join(' · ') || '—';
    }
    const snippet = c.text || '';
    body.innerHTML = snippet
      ? `<mark>${this._escape(snippet)}</mark>`
      : '<span class="muted">Текст фрагмента не передан</span>';
    drawer.classList.remove('hidden');
    body.scrollTop = 0;
  },

  closeCitation() {
    document.getElementById('cite-drawer')?.classList.add('hidden');
  },

  _showTrace(traceEl, mode, data) {
    if (!traceEl || !data.trace_id) return;
    const q = mode === 'agent'
      ? `run_id=${encodeURIComponent(data.trace_id)}`
      : `trace_id=${encodeURIComponent(data.trace_id)}`;
    const modeLabel = this._modeRu(data.retrieval_mode || mode);
    const ground = data.grounding_ok == null
      ? ''
      : (data.grounding_ok ? ' · опора на источники: да' : ' · опора на источники: нет');
    traceEl.classList.remove('hidden');
    traceEl.innerHTML =
      `Трасса: <a href="/trace?${q}">${this._escape(data.trace_id)}</a> · ${this._escape(modeLabel)}${ground}`;
  },

  _showCitations(sourcesEl, citations) {
    if (!sourcesEl) return;
    this._lastCitations = citations || [];
    sourcesEl.innerHTML = (citations || []).map((c) => {
      const kind = c.kind === 'web' || c.url ? 'web' : 'kb';
      const loc = c.url
        ? this._escape(c.url)
        : (c.section_path ? this._escape(c.section_path) : '');
      const scoreBit = kind === 'kb' && c.score != null ? ` · релевантность ${c.score}` : '';
      return `
      <div class="source" role="button" tabindex="0" data-cite="${c.n}" title="Показать цитируемый фрагмент">
        <div class="meta">[${c.n}] ${kind === 'web' ? '[web] ' : ''}${this._escape(c.title)}${scoreBit}${loc ? ' · ' + loc : ''}</div>
        <div class="muted">${this._escape((c.text || '').slice(0, 180))}${(c.text || '').length > 180 ? '…' : ''}</div>
      </div>`;
    }).join('');
    sourcesEl.querySelectorAll('[data-cite]').forEach((el) => {
      const open = () => this.openCitation(Number(el.getAttribute('data-cite')));
      el.addEventListener('click', open);
      el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      });
    });
  },

  _chatPayload() {
    return {
      query: document.getElementById('query').value.trim(),
      top_k: Number(document.getElementById('top_k').value || 5),
      mode: document.getElementById('mode')?.value || 'rag',
      space: document.getElementById('space')?.value || 'all',
    };
  },

  async chat() {
    const payload = this._chatPayload();
    const { mode } = payload;
    if (!payload.query) return;
    this._beginRun(mode === 'agent' ? 'Агент работает…' : 'Ищем и генерируем ответ…');
    this._addStep('Запуск', mode === 'agent' ? 'мультиагентный режим' : 'быстрый RAG', { active: true });
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        this._setStatus(data.detail || 'Ошибка', { error: true });
        this._showAnswer(data.detail || 'Ошибка');
        return;
      }
      if (data.retrieval_mode) {
        this._addStep('Поиск', this._modeRu(data.retrieval_mode));
      }
      this._addStep('Ответ готов', data.model ? `модель: ${data.model}` : '');
      this._showAnswer(data.answer);
      this._showTrace(this._els().trace, mode, data);
      this._showCitations(this._els().sources, data.citations);
      this._setStatus('Готово', { done: true });
    } catch (err) {
      this._setStatus(String(err), { error: true });
      this._showAnswer(String(err));
    }
  },

  async chatStream() {
    const payload = this._chatPayload();
    const { mode } = payload;
    if (!payload.query) return;
    this._beginRun(mode === 'agent' ? 'Агент запускается…' : 'Запускаем поиск…');
    this._addStep('Подключение', 'потоковый режим', { active: true });
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok || !res.body) {
        this._setStatus('Ошибка потока', { error: true });
        this._showAnswer('Не удалось открыть поток ответа');
        return;
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let event = 'message';
      let answerStarted = false;
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

          if (event === 'status') {
            this._setStatus(data.text || 'Работаем…');
            if (data.text) this._addStep('Статус', data.text, { active: true });
          }
          if (event === 'start') {
            this._setStatus(mode === 'agent' ? 'Агент планирует…' : 'Ищем по базе…');
          }
          if (event === 'plan') {
            this._setStatus('План готов, выполняем шаги…');
            this._addStep(this._stepTitle({ kind: 'plan', output: data }), this._stepDetail({ kind: 'plan', output: data }));
          }
          if (event === 'step') {
            this._setStatus(this._stepTitle(data) + '…');
            this._addStep(this._stepTitle(data), this._stepDetail(data), { active: true });
          }
          if (event === 'retrieval') {
            const docs = (data.docs || []).slice(0, 3).join(', ');
            const detail = `${this._modeRu(data.mode)} · найдено: ${data.n}` +
              (docs ? ` · примеры: ${docs}` : '');
            this._setStatus('Контекст собран, генерируем ответ…');
            this._addStep('Поиск по базе', detail, { active: true });
          }
          if (event === 'token') {
            if (!answerStarted) {
              answerStarted = true;
              this._setStatus('Печатаем ответ…');
              this._addStep('Генерация', 'поток токенов');
              this._showAnswer('');
            }
            this._appendAnswer(data.text || '');
          }
          if (event === 'final') {
            this._showAnswer(data.answer || this._els().answer?.textContent || '');
            this._showTrace(this._els().trace, mode, data);
            this._showCitations(this._els().sources, data.citations);
            this._setStatus('Готово', { done: true });
            this._addStep('Готово', data.model ? `модель: ${data.model}` : '');
          }
          if (event === 'error') {
            this._setStatus(data.detail || 'Ошибка', { error: true });
            this._showAnswer(data.detail || 'Ошибка');
          }
        }
      }
      const { panel } = this._els();
      if (panel?.classList.contains('running')) {
        this._setStatus('Готово', { done: true });
      }
    } catch (err) {
      this._setStatus(String(err), { error: true });
      this._showAnswer(String(err));
    }
  },

  async searchOnly() {
    const payload = this._chatPayload();
    if (!payload.query) return;
    this._beginRun('Только поиск по базе…');
    this._addStep('Поиск', 'без генерации ответа', { active: true });
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: payload.query,
          top_k: payload.top_k,
          space: payload.space,
        }),
      });
      const data = await res.json();
      const { answer, sources } = this._els();
      if (!res.ok) {
        this._setStatus(data.detail || 'Ошибка', { error: true });
        return;
      }
      if (!(data.results || []).length) {
        this._addStep('Результат', 'ничего не найдено');
        if (sources) {
          sources.innerHTML =
            `<p class="muted">Пусто (${this._escape(this._modeRu(data.retrieval_mode))}). ` +
            `Для демо-корпуса выберите пространство «Общее» или «Все».</p>`;
        }
        this._setStatus('Ничего не найдено', { done: true });
        if (answer) answer.classList.add('hidden');
        return;
      }
      this._addStep(
        'Найдено',
        `${data.results.length} фрагм. · ${this._modeRu(data.retrieval_mode)}`,
      );
      if (answer) answer.classList.add('hidden');
      if (sources) {
        sources.innerHTML = (data.results || []).map((r, i) => `
          <div class="source">
            <div class="meta">[${i + 1}] ${this._escape(r.document_name)} · релевантность ${r.score}</div>
            <div>${this._escape(r.text.slice(0, 400))}${r.text.length > 400 ? '…' : ''}</div>
          </div>`).join('');
      }
      this._setStatus(`Найдено фрагментов: ${data.results.length}`, { done: true });
    } catch (err) {
      this._setStatus(String(err), { error: true });
    }
  },

  async uploadFile() {
    const file = document.getElementById('file').files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    const space = document.getElementById('upload-space')?.value;
    if (space) fd.append('space', space);
    const res = await fetch('/api/documents/upload', { method: 'POST', body: fd });
    const data = await res.json();
    const el = document.getElementById('job-status');
    if (!res.ok) { el.textContent = data.detail || 'Ошибка'; return; }
    el.textContent = `Задача #${data.job_id} для документа #${data.document_id}`;
    this.pollJob(data.job_id);
  },

  async uploadUrl() {
    const url = document.getElementById('url').value.trim();
    const space = document.getElementById('upload-space')?.value || 'personal';
    const res = await fetch('/api/documents/from-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, space }),
    });
    const data = await res.json();
    const el = document.getElementById('job-status');
    if (!res.ok) { el.textContent = data.detail || 'Ошибка'; return; }
    el.textContent = `Задача #${data.job_id} для документа #${data.document_id}`;
    this.pollJob(data.job_id);
  },

  async pollJob(jobId) {
    const el = document.getElementById('job-status');
    const statusRu = {
      pending: 'в очереди',
      running: 'выполняется',
      completed: 'готово',
      failed: 'ошибка',
    };
    const tick = async () => {
      const res = await fetch(`/api/jobs/${jobId}`);
      const j = await res.json();
      const st = statusRu[j.status] || j.status;
      el.innerHTML = `<span class="spinner inline"></span> задача #${j.id}: ${st} · ${j.progress}%` +
        (j.current_step ? ` · ${j.current_step}` : '');
      if (j.status === 'completed' || j.status === 'failed') {
        el.textContent = `задача #${j.id}: ${st}` + (j.error_message ? ` — ${j.error_message}` : '');
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
    const statusRu = {
      pending: 'ожидает',
      indexing: 'индексация',
      indexed: 'проиндексирован',
      error: 'ошибка',
    };
    const spaceRu = { personal: 'личное', project: 'проект', shared: 'общее' };
    box.innerHTML = (data.items || []).map((d) => `
      <div class="doc-row">
        <div>
          <strong>${this._escape(d.name)}</strong>
          <span class="badge ${d.status === 'indexed' ? 'ok' : d.status === 'error' ? 'err' : ''}">${statusRu[d.status] || d.status}</span>
          <div class="muted">${d.num_chunks} фрагм. · ${spaceRu[d.space] || d.space}</div>
        </div>
        <div class="row">
          <button class="btn btn-ghost" onclick="PrismApp.previewChunks(${d.id})">Фрагменты</button>
          <button class="btn btn-ghost" onclick="PrismApp.deleteDoc(${d.id})">Удалить</button>
        </div>
      </div>`).join('') || '<p class="muted">Пока пусто</p>';
  },

  async previewChunks(id) {
    const box = document.getElementById('chunk-preview');
    if (!box) return;
    box.innerHTML = '<span class="spinner inline"></span> Загрузка…';
    const res = await fetch(`/api/documents/${id}/chunks?limit=20`);
    const data = await res.json();
    if (!res.ok) {
      box.textContent = data.detail || 'Ошибка';
      return;
    }
    box.innerHTML = `<h3>${this._escape(data.name)} · всего ${data.total}</h3>` +
      (data.items || []).map((c) => `
        <div class="doc-row">
          <div>
            <strong>#${c.chunk_index}</strong>
            <span class="muted">${this._escape(c.section_path || '')} · ${c.token_count || 0} ток.</span>
            <pre class="answer" style="white-space:pre-wrap;margin:.4rem 0 0">${this._escape(c.text || '')}</pre>
          </div>
        </div>`).join('');
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
      ? 'Сохранено'
      : (data.detail || 'Ошибка');
  },
};
