const PrismApp = {
  _lastCitations: [],
  /** @type {{role:'user'|'assistant', content:string, citations?:any[], details_html?:string}[]} */
  _thread: [],
  /** @type {number|null} */
  _dialogId: null,
  _dialogs: [],

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

  /** Lightweight markdown → safe HTML (answers, journal). Escapes first. */
  _renderMarkdown(src) {
    let text = String(src ?? '').replace(/\r\n/g, '\n');
    const blocks = [];
    text = text.replace(/```([\w-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const i = blocks.length;
      blocks.push(
        `<pre class="md-code"><code${lang ? ` data-lang="${this._escapeAttr(lang)}"` : ''}>${this._escape(code.replace(/\n$/, ''))}</code></pre>`
      );
      return `\u0000BLK${i}\u0000`;
    });

    const renderTable = (chunk) => {
      const lines = chunk.trim().split('\n').filter(Boolean);
      if (lines.length < 2 || !/^\|/.test(lines[0])) return null;
      const split = (line) => line.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const head = split(lines[0]);
      if (!/^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$/.test(lines[1])) return null;
      const rows = lines.slice(2).map(split);
      let html = '<table class="md-table"><thead><tr>';
      head.forEach((h) => { html += `<th>${this._inlineMd(h)}</th>`; });
      html += '</tr></thead><tbody>';
      rows.forEach((r) => {
        html += '<tr>';
        head.forEach((_, i) => { html += `<td>${this._inlineMd(r[i] || '')}</td>`; });
        html += '</tr>';
      });
      html += '</tbody></table>';
      return html;
    };

    const parts = text.split(/\n{2,}/);
    const htmlParts = parts.map((part) => {
      const trimmed = part.trim();
      if (!trimmed) return '';
      if (/^\u0000BLK\d+\u0000$/.test(trimmed)) return trimmed;
      const table = renderTable(trimmed);
      if (table) return table;
      if (/^#{1,3} /.test(trimmed)) {
        return trimmed.split('\n').map((line) => {
          const m = /^(#{1,3})\s+(.+)$/.exec(line);
          if (!m) return `<p>${this._inlineMd(line)}</p>`;
          const lvl = m[1].length;
          return `<h${lvl}>${this._inlineMd(m[2])}</h${lvl}>`;
        }).join('');
      }
      if ((/^[-*]\s+/m.test(trimmed) || /^\d+\.\s+/m.test(trimmed))
          && trimmed.split('\n').every((l) => !l.trim() || /^([-*]|\d+\.)\s+/.test(l))) {
        const items = trimmed.split('\n').filter(Boolean).map((l) => {
          const m = /^[-*]\s+(.+)$/.exec(l) || /^\d+\.\s+(.+)$/.exec(l);
          return `<li>${this._inlineMd(m ? m[1] : l)}</li>`;
        });
        const ordered = /^\d+\./.test(trimmed);
        return ordered ? `<ol>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`;
      }
      if (/^>\s?/m.test(trimmed)) {
        const body = trimmed.split('\n').map((l) => l.replace(/^>\s?/, '')).join(' ');
        return `<blockquote>${this._inlineMd(body)}</blockquote>`;
      }
      return `<p>${trimmed.split('\n').map((l) => this._inlineMd(l)).join('<br>')}</p>`;
    });

    let html = htmlParts.join('\n');
    html = html.replace(/\u0000BLK(\d+)\u0000/g, (_, i) => blocks[Number(i)] || '');
    return html;
  },

  _inlineMd(s) {
    let t = this._escape(s);
    t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
    t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    t = t.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
    t = t.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    t = t.replace(/\[(\d+)\]/g, '<span class="cite-ref">[$1]</span>');
    return t;
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
      if (tool === 'web_search') {
        const prov = out && typeof out === 'object' ? out.provider : null;
        return prov && String(prov).startsWith('mock')
          ? `веб-поиск · mock (${prov})`
          : 'веб-поиск';
      }
      if (tool === 'build_comparison_table') return 'сводная таблица';
      if (tool === 'distill_run') return 'запись выводов в память';
      return tool || 'вызов инструмента';
    }
    if (kind === 'delegation') {
      const roles = (step.input?.roles || []).join(', ');
      const phase = step.input?.phase === 'start' ? 'старт'
        : step.input?.phase === 'done' ? 'готово' : '';
      const bits = [];
      if (phase) bits.push(phase);
      if (roles) bits.push(`роли: ${roles}`);
      return bits.join(' · ') || 'запуск субагентов';
    }
    if (kind === 'synthesize') {
      return step.input?.revision ? 'повторная сборка после критика' : 'финальная формулировка';
    }
    if (typeof step.output === 'string') return step.output.slice(0, 160);
    return '';
  },

  _isChatPage() {
    return !!document.getElementById('chat-page');
  },

  _els() {
    return {
      panel: document.getElementById('run-panel'),
      status: document.getElementById('run-status'),
      statusText: document.querySelector('#run-status .run-status-text'),
      steps: document.getElementById('run-steps'),
      note: document.getElementById('answer-note'),
      answer: document.getElementById('answer'),
      sources: document.getElementById('sources'),
      trace: document.getElementById('trace-link'),
      ask: document.getElementById('btn-ask'),
      live: document.getElementById('live-turn'),
      liveUser: document.getElementById('live-user'),
      empty: document.getElementById('chat-empty'),
      scroll: document.getElementById('chat-scroll'),
      details: document.getElementById('run-details'),
      clearBtn: document.getElementById('btn-clear-thread'),
    };
  },

  _scrollChatBottom() {
    const { scroll } = this._els();
    if (scroll) scroll.scrollTop = scroll.scrollHeight;
  },

  _setBusy(busy) {
    const { ask } = this._els();
    if (ask) ask.disabled = busy;
  },

  _beginRun(label, { query = '' } = {}) {
    const {
      panel, status, statusText, steps, note, answer, sources, trace,
      live, liveUser, empty, details, clearBtn,
    } = this._els();
    if (this._isChatPage()) {
      if (empty) empty.classList.add('hidden');
      if (live) live.classList.remove('hidden');
      if (liveUser) liveUser.textContent = query;
      if (clearBtn) clearBtn.classList.remove('hidden');
      if (details) details.open = false;
    }
    if (!panel) {
      this._setBusy(true);
      return;
    }
    panel.classList.remove('hidden');
    panel.classList.add('running');
    if (status) status.classList.remove('done', 'error');
    if (statusText) statusText.textContent = label;
    if (steps) steps.innerHTML = '';
    if (note) {
      note.className = 'answer-note hidden';
      note.textContent = '';
    }
    if (answer) {
      answer.classList.add('hidden');
      answer.classList.remove('is-provisional');
      answer.textContent = '';
    }
    if (sources) sources.innerHTML = '';
    if (trace) {
      trace.classList.add('hidden');
      trace.innerHTML = '';
    }
    this._setBusy(true);
    this._scrollChatBottom();
  },

  _setAnswerNote(text, kind = '') {
    const { note } = this._els();
    if (!note) return;
    if (!text) {
      note.className = 'answer-note hidden';
      note.textContent = '';
      return;
    }
    note.className = 'answer-note' + (kind ? ` ${kind}` : '');
    note.textContent = text;
  },

  _markAnswerProvisional(on) {
    const { answer } = this._els();
    if (!answer) return;
    answer.classList.toggle('is-provisional', !!on);
    if (on) {
      this._setAnswerNote(
        'Черновик: текст идёт с модели. Проверка опоры на источники — после завершения.',
        'is-provisional',
      );
    }
  },

  _applyGrounding(_data) {
    // Grounding still runs server-side (journal/eval); don't scare the chat UI.
    this._markAnswerProvisional(false);
    this._setAnswerNote('');
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
    if (!this._isChatPage()) {
      li.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
    return li;
  },

  _renderAnswerHtml(text) {
    let html = this._renderMarkdown(text || '');
    // Markdown turns [n] into spans; make them clickable cite links.
    html = html.replace(
      /<span class="cite-ref">\[(\d+)\]<\/span>/g,
      '<a href="#cite-$1" class="cite-ref" data-cite="$1" title="Открыть источник [$1]">[$1]</a>',
    );
    // Any leftover bare [n] outside inline md paths.
    html = html.replace(
      /(?<!["=])\[(\d+)\](?!:)/g,
      '<a href="#cite-$1" class="cite-ref" data-cite="$1" title="Открыть источник [$1]">[$1]</a>',
    );
    return html;
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
    answer.classList.add('md-body');
    answer.innerHTML = this._renderAnswerHtml(text);
    this._bindAnswerCitationClicks(answer);
    this._scrollChatBottom();
  },

  _appendAnswer(text) {
    const { answer } = this._els();
    if (!answer) return;
    answer.classList.remove('hidden');
    answer.classList.remove('md-body');
    // Live tokens: plain text; final re-renders as markdown.
    answer.textContent += text || '';
    this._scrollChatBottom();
  },

  _findCitation(n, pool) {
    const want = Number(n);
    const from = (list) => (list || []).find((x) => Number(x.n) === want);
    let c = from(pool) || from(this._lastCitations);
    if (c) return c;
    for (const t of this._thread || []) {
      c = from(t.citations);
      if (c) return c;
    }
    return null;
  },

  openCitation(n, pool) {
    const c = this._findCitation(n, pool);
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
    this._lastCitations = pool || this._lastCitations || [c];
    const kind = c.kind === 'web' || c.url ? 'web' : 'kb';
    title.textContent = `[${c.n}] ${c.title || c.document_name || (kind === 'web' ? 'веб' : 'документ')}`;
    const bits = [];
    if (kind === 'web') bits.push('веб');
    if (c.page != null) bits.push(`стр. ${c.page}`);
    if (c.chunk_id) bits.push(`фрагмент #${c.chunk_id}`);
    if (c.score != null && kind === 'kb') bits.push(`релевантность ${c.score}`);
    const links = [];
    if (c.doc_id) {
      links.push(
        `<a href="/upload/${encodeURIComponent(c.doc_id)}?chunk=${encodeURIComponent(c.chunk_id || '')}" target="_blank" rel="noopener noreferrer">открыть документ</a>`,
      );
    }
    if (c.url) {
      links.push(
        `<a href="${this._escapeAttr(c.url)}" target="_blank" rel="noopener noreferrer">${this._escape(c.url)}</a>`,
      );
    }
    if (c.section_path && kind === 'kb') bits.push(c.section_path);
    meta.innerHTML = [
      bits.length ? this._escape(bits.join(' · ')) : '',
      links.join(' · '),
    ].filter(Boolean).join(' · ') || '—';
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
    const href = mode === 'agent'
      ? `/trace/run/${encodeURIComponent(data.trace_id)}`
      : `/trace/log/${encodeURIComponent(data.trace_id)}`;
    const modeLabel = this._modeRu(data.retrieval_mode || mode);
    traceEl.classList.remove('hidden');
    traceEl.innerHTML = `
      <div class="trace-meta">
        <span class="muted">${this._escape(modeLabel)}</span>
      </div>
      <a class="btn btn-ghost btn-sm" href="${href}">Открыть в журнале</a>`;
  },

  _sourceDocHref(c) {
    if (!c) return '';
    if (c.doc_id) {
      const chunk = c.chunk_id != null && c.chunk_id !== '' ? `?chunk=${encodeURIComponent(c.chunk_id)}` : '';
      return `/upload/${encodeURIComponent(c.doc_id)}${chunk}`;
    }
    if (c.url) return String(c.url);
    return '';
  },

  _sourcesDetailsHtml(citations) {
    const list = citations || [];
    if (!list.length) return '';
    const items = list.map((c) => {
      const kind = c.kind === 'web' || c.url ? 'web' : 'kb';
      const title = c.title || c.document_name || 'источник';
      const label = `[${c.n}] ${kind === 'web' ? '[web] ' : ''}${this._escape(title)}`;
      const href = this._sourceDocHref(c);
      const meta = href
        ? `<a class="meta source-doc" href="${this._escapeAttr(href)}" target="_blank" rel="noopener noreferrer" title="Открыть документ с подсветкой">${label}</a>`
        : `<button type="button" class="meta source-doc source-doc-btn" data-cite="${c.n}" title="Показать фрагмент">${label}</button>`;
      return `<div class="source" id="cite-${c.n}">${meta}` +
        `<button type="button" class="source-peek" data-cite="${c.n}" title="Показать фрагмент">фрагмент</button>` +
        `</div>`;
    }).join('');
    return `<details class="sources-details chat-sources">` +
      `<summary class="sources-summary">Источники <span class="muted">${list.length}</span></summary>` +
      `<div class="sources">${items}</div></details>`;
  },

  _bindSourceClicks(root, citations) {
    if (!root) return;
    root.querySelectorAll('[data-cite]').forEach((el) => {
      const open = () => this.openCitation(Number(el.getAttribute('data-cite')), citations);
      el.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        open();
      });
      el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      });
    });
  },

  _showCitations(sourcesEl, citations) {
    if (!sourcesEl) return;
    this._lastCitations = citations || [];
    sourcesEl.innerHTML = this._sourcesDetailsHtml(citations);
    this._bindSourceClicks(sourcesEl, citations);
  },

  _chatPayload() {
    // top_k берётся из Настроек на сервере (не дублируем контрол на форме)
    const mode = document.getElementById('mode')?.value || 'rag';
    return {
      query: document.getElementById('query').value.trim(),
      mode: mode === 'search' ? 'rag' : mode,
      space: document.getElementById('space')?.value || 'all',
      history: this._thread.filter((t) => !t.approval).slice(-8),
    };
  },

  renderThread() {
    const el = document.getElementById('thread');
    const { empty, live } = this._els();
    const has = this._thread.length > 0;
    const liveOn = live && !live.classList.contains('hidden');
    if (empty) empty.classList.toggle('hidden', has || liveOn);
    if (!el) return;
    if (!has) {
      el.innerHTML = '';
      this._highlightActiveDialog();
      return;
    }
    el.innerHTML = this._thread.map((t) => {
      const role = t.role === 'user' ? 'Вы' : 'Призма';
      const body = t.role === 'assistant'
        ? this._renderAnswerHtml(t.content)
        : this._escape(t.content);
      const cites = (t.role === 'assistant' && t.citations && t.citations.length)
        ? this._sourcesDetailsHtml(t.citations)
        : '';
      const details = (t.role === 'assistant' && t.details_html) ? t.details_html : '';
      const gate = (t.role === 'assistant' && t.approval) ? this._approvalCardHtml(t) : '';
      const bodyClass = t.role === 'assistant' ? 'thread-body md-body' : 'thread-body';
      return `<article class="thread-turn is-${this._escapeAttr(t.role)}" data-turn-cites="${t.role === 'assistant' && t.citations ? '1' : '0'}">` +
        `<div class="thread-role">${role}</div>` +
        `<div class="${bodyClass}">${body}</div>${gate}${cites}${details}</article>`;
    }).join('');
    el.querySelectorAll('article.thread-turn').forEach((art, idx) => {
      const turn = this._thread[idx];
      const cites = turn?.citations || null;
      art.querySelectorAll('[data-cite]').forEach((node) => {
        node.addEventListener('click', (e) => {
          e.preventDefault();
          this.openCitation(Number(node.getAttribute('data-cite')), cites);
        });
      });
    });
    el.querySelectorAll('[data-approve], [data-deny]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-approve') || btn.getAttribute('data-deny');
        this.decideApproval(id, btn.hasAttribute('data-approve'));
      });
    });
    this._highlightActiveDialog();
    this._scrollChatBottom();
    this._setComposerLocked(this._thread.some((t) => t.approval));
  },

  clearThread() {
    this._dialogId = null;
    this._thread = [];
    const { live, answer, sources, panel, empty } = this._els();
    if (live) live.classList.add('hidden');
    if (answer) {
      answer.classList.add('hidden');
      answer.textContent = '';
    }
    if (sources) sources.innerHTML = '';
    if (panel) panel.classList.add('hidden');
    if (empty) empty.classList.remove('hidden');
    this.renderThread();
    this._highlightActiveDialog();
    const q = document.getElementById('query');
    if (q) {
      q.value = '';
      q.focus();
    }
  },

  async loadDialogList() {
    if (!this._isChatPage()) return;
    const box = document.getElementById('dialog-list');
    if (!box) return;
    try {
      const res = await fetch('/api/dialogs');
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'list failed');
      this._dialogs = data.items || [];
    } catch {
      this._dialogs = [];
    }
    this._renderDialogList();
  },

  _renderDialogList() {
    const box = document.getElementById('dialog-list');
    if (!box) return;
    if (!this._dialogs.length) {
      box.innerHTML = '<p class="muted dialog-list-empty">Пока пусто</p>';
      return;
    }
    box.innerHTML = this._dialogs.map((d) => {
      const active = this._dialogId === d.id ? ' is-active' : '';
      return `<div class="dialog-item${active}" data-dialog-id="${d.id}">` +
        `<button type="button" class="dialog-item-open" data-dialog-open="${d.id}">` +
        `<span class="dialog-item-title">${this._escape(d.title || 'Диалог')}</span>` +
        `<span class="dialog-item-meta muted">${d.n_messages || 0} сообщ.</span>` +
        `</button>` +
        `<button type="button" class="dialog-item-del" data-dialog-del="${d.id}" title="Удалить" aria-label="Удалить">×</button>` +
        `</div>`;
    }).join('');
    box.querySelectorAll('[data-dialog-open]').forEach((btn) => {
      btn.addEventListener('click', () => this.openDialog(Number(btn.getAttribute('data-dialog-open'))));
    });
    box.querySelectorAll('[data-dialog-del]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteDialog(Number(btn.getAttribute('data-dialog-del')));
      });
    });
  },

  _highlightActiveDialog() {
    document.querySelectorAll('.dialog-item').forEach((el) => {
      el.classList.toggle('is-active', Number(el.getAttribute('data-dialog-id')) === this._dialogId);
    });
  },

  async openDialog(id) {
    if (!id) return;
    try {
      const res = await fetch(`/api/dialogs/${id}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'load failed');
      this._dialogId = data.id;
      this._thread = (data.messages || []).map((m) => ({
        role: m.role,
        content: m.content,
        citations: m.citations || null,
        approval: m.approval || null,
      }));
      const { live, panel, answer, sources } = this._els();
      if (live) live.classList.add('hidden');
      if (panel) panel.classList.add('hidden');
      if (answer) {
        answer.classList.add('hidden');
        answer.textContent = '';
      }
      if (sources) sources.innerHTML = '';
      const meta = data.meta || {};
      if (meta.mode) {
        const modeSel = document.getElementById('mode');
        const radio = document.querySelector(`input[name="mode"][value="${meta.mode}"]`);
        if (modeSel) modeSel.value = meta.mode;
        if (radio) radio.checked = true;
      }
      if (meta.space) {
        const spaceSel = document.getElementById('space');
        const radio = document.querySelector(`input[name="space"][value="${meta.space}"]`);
        if (spaceSel) spaceSel.value = meta.space;
        if (radio) radio.checked = true;
      }
      this.renderThread();
      this._highlightActiveDialog();
    } catch (err) {
      console.warn(err);
    }
  },

  async deleteDialog(id) {
    const ok = await this.confirm({
      title: 'Удалить диалог?',
      body: 'История переписки будет удалена без восстановления.',
      confirmLabel: 'Удалить',
      danger: true,
    });
    if (!ok) return;
    try {
      const res = await fetch(`/api/dialogs/${id}`, { method: 'DELETE' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'delete failed');
      }
      if (this._dialogId === id) this.clearThread();
      await this.loadDialogList();
    } catch (err) {
      console.warn(err);
    }
  },

  async saveDialog() {
    if (!this._isChatPage() || !this._thread.length) return;
    const messages = this._thread.map((t) => ({
      role: t.role,
      content: t.content,
      ...(t.citations ? { citations: t.citations } : {}),
      ...(t.approval ? { approval: t.approval } : {}),
    }));
    const body = {
      id: this._dialogId,
      messages,
      meta: {
        mode: document.getElementById('mode')?.value || 'rag',
        space: document.getElementById('space')?.value || 'shared',
      },
    };
    try {
      const res = await fetch('/api/dialogs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'save failed');
      this._dialogId = data.id;
      await this.loadDialogList();
    } catch (err) {
      console.warn(err);
    }
  },

  _commitTurn(query, answer, extra = {}) {
    const q = String(query || '').trim();
    const a = String(answer || '').trim();
    if (!q || !a) return;
    const detailsHtml = this._detailsHtml();
    this._thread.push({ role: 'user', content: q });
    this._thread.push({
      role: 'assistant',
      content: a,
      citations: extra.citations || null,
      details_html: detailsHtml,
    });
    if (this._thread.length > 16) this._thread = this._thread.slice(-16);
    const { live, panel } = this._els();
    if (live) live.classList.add('hidden');
    if (panel) panel.classList.add('hidden');
    this.renderThread();
    const input = document.getElementById('query');
    if (input) input.value = '';
    this.saveDialog();
  },

  async chat() {
    return this.chatStream();
  },

  _detailsHtml() {
    if (!this._isChatPage()) return '';
    const { steps, trace } = this._els();
    const stepsClone = steps && steps.children.length ? steps.outerHTML : '';
    const traceClone = (trace && !trace.classList.contains('hidden') ? trace.outerHTML : '') || '';
    if (!stepsClone && !traceClone) return '';
    return `<details class="run-details thread-details">` +
      `<summary class="run-summary"><span>Ход ответа</span><span class="run-summary-hint muted">шаги</span></summary>` +
      `${stepsClone}${traceClone}</details>`;
  },

  _toolRu(code) {
    const names = {
      web_search: 'поиск в интернете',
      fetch_page: 'чтение страницы',
      python_sandbox: 'песочница Python',
      memory_rollback: 'откат памяти',
    };
    return names[code] || code || 'действие';
  },

  _approvalLead(approval) {
    return `Ждёт одобрения: ${this._toolRu(approval && approval.tool)}`;
  },

  _approvalArgsText(approval) {
    const args = (approval && approval.args) || {};
    if (args.query) return String(args.query);
    if (args.url) return String(args.url);
    try {
      return JSON.stringify(args, null, 2);
    } catch {
      return '';
    }
  },

  _approvalCardHtml(turn) {
    const approval = turn.approval || {};
    if (turn._deciding) {
      return `<div class="approval-card"><p class="approval-title"><span class="spinner inline"></span> Продолжаем тот же запуск…</p></div>`;
    }
    const args = this._approvalArgsText(approval);
    const argsHtml = args
      ? `<pre class="approval-args">${this._escape(args)}</pre>`
      : '';
    return `<div class="approval-card">` +
      `<p class="approval-title">Нужно разрешение: ${this._escape(this._toolRu(approval.tool))}</p>` +
      argsHtml +
      `<div class="row">` +
      `<button type="button" class="btn" data-approve="${this._escapeAttr(approval.id)}">Одобрить</button>` +
      `<button type="button" class="btn btn-ghost" data-deny="${this._escapeAttr(approval.id)}">Отклонить</button>` +
      `</div></div>`;
  },

  _setComposerLocked(locked) {
    const form = document.getElementById('chat-form');
    if (!form) return;
    form.querySelectorAll('textarea, button, input, select').forEach((el) => {
      if (el.classList.contains('sr-only')) return;
      el.disabled = !!locked;
    });
    const area = document.getElementById('query');
    if (area) {
      area.placeholder = locked
        ? 'Сначала одобрите или отклоните действие агента…'
        : 'Спросите или уточните…';
    }
  },

  _holdForApproval(query, data) {
    const approval = data.approval;
    const q = String(query || '').trim();
    if (!q || !approval || !approval.id) return;
    this._thread.push({ role: 'user', content: q });
    this._thread.push({
      role: 'assistant',
      content: this._approvalLead(approval),
      approval,
      details_html: this._detailsHtml(),
    });
    if (this._thread.length > 16) this._thread = this._thread.slice(-16);
    const { live, panel } = this._els();
    if (live) live.classList.add('hidden');
    if (panel) {
      panel.classList.add('hidden');
      panel.classList.remove('running');
    }
    this._setBusy(false);
    this.renderThread();
    this.saveDialog();
  },

  async decideApproval(id, allow) {
    const msg = this._thread.find((t) => t.approval && String(t.approval.id) === String(id));
    if (!msg || msg._deciding) return;
    msg._deciding = true;
    this.renderThread();
    try {
      const res = await fetch(`/api/approvals/${encodeURIComponent(id)}/${allow ? 'approve' : 'deny'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (res.status === 404 || res.status === 410) {
          msg.content = data.detail || 'Запрос на одобрение уже не действует.';
          msg.approval = null;
        } else {
          msg.content = data.detail || 'Не удалось обработать решение. Можно повторить.';
        }
        return;
      }
      if (!allow) {
        msg.content = 'Действие отклонено. Запуск остановлен.';
        msg.approval = null;
        return;
      }
      if (data.resume_error) {
        msg.content = `Одобрено, но запуск не продолжился: ${data.resume_error}`;
        msg.approval = null;
        return;
      }
      const run = data.run || {};
      if (run.state === 'awaiting_approval' && run.approval && run.approval.id) {
        msg.approval = run.approval;
        msg.content = this._approvalLead(run.approval);
        return;
      }
      msg.approval = null;
      msg.content = run.answer || (run.state === 'failed' ? 'Ошибка выполнения' : '');
      if (run.citations && run.citations.length) msg.citations = run.citations;
    } catch (err) {
      msg.content = `${err}. Можно повторить.`;
    } finally {
      msg._deciding = false;
      this.renderThread();
      this.saveDialog();
    }
  },

  async chatStream() {
    const payload = this._chatPayload();
    const { mode } = payload;
    if (!payload.query) return;
    if (this._thread.some((t) => t.approval)) return;
    const q = payload.query;
    const input = document.getElementById('query');
    if (input && this._isChatPage()) input.value = '';
    this._beginRun(
      mode === 'agent' ? 'Агент запускается…' : 'Ищем по базе…',
      { query: q },
    );
    this._addStep('Подключение', mode === 'agent' ? 'агент' : 'RAG', { active: true });
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, query: q }),
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
            this._setStatus('Генерируем ответ…');
            this._addStep('Поиск по базе', detail, { active: true });
          }
          if (event === 'chunk' || event === 'token') {
            if (!answerStarted) {
              answerStarted = true;
              this._setStatus('Печатаем ответ…');
              this._addStep('Генерация', 'поток от модели');
              this._showAnswer('');
            }
            this._appendAnswer(data.text || '');
          }
          if (event === 'final') {
            if (data.state === 'awaiting_approval' && data.approval && data.approval.id) {
              this._holdForApproval(q, data);
              continue;
            }
            const answerText = data.answer || this._els().answer?.textContent || '';
            this._showAnswer(answerText);
            this._applyGrounding(data);
            this._showTrace(this._els().trace, mode, data);
            this._showCitations(this._els().sources, data.citations);
            this._addStep('Готово', data.model ? `модель: ${data.model}` : '');
            this._setStatus('Готово', { done: true });
            this._commitTurn(q, answerText, { citations: data.citations || [] });
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
          history: payload.history || [],
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
        const fake = (data.results || []).map((r, i) => ({
          n: i + 1,
          title: r.document_name,
          document_name: r.document_name,
          text: r.text,
          score: r.score,
          doc_id: r.document_id,
          chunk_id: r.chunk_id,
        }));
        this._showCitations(sources, fake);
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

  _docsPage: 1,
  _docsPageSize: 10,

  async loadDocuments(page, pageSize) {
    const box = document.getElementById('docs');
    if (!box) return 1;
    if (page != null) this._docsPage = page;
    if (pageSize != null) this._docsPageSize = pageSize;
    const p = this._docsPage;
    const ps = this._docsPageSize;
    const res = await fetch(`/api/documents?page=${p}&page_size=${ps}`);
    const data = await res.json();
    const statusRu = {
      pending: 'ожидает',
      indexing: 'индексация',
      indexed: 'проиндексирован',
      error: 'ошибка',
    };
    const spaceRu = { personal: 'личное', project: 'проект', shared: 'общее' };
    const pages = data.pages || 1;
    this._docsPage = data.page || p;
    box.innerHTML = (data.items || []).map((d) => `
      <div class="doc-row">
        <div>
          <strong>${this._escape(d.name)}</strong>
          <span class="badge ${d.status === 'indexed' ? 'ok' : d.status === 'error' ? 'err' : ''}">${statusRu[d.status] || d.status}</span>
          <div class="muted">${d.num_chunks} фрагм. · ${spaceRu[d.space] || d.space}</div>
        </div>
        <div class="row doc-row-actions">
          <a class="btn btn-ghost btn-sm" href="/upload/${d.id}">Фрагменты</a>
          <button type="button" class="btn btn-ghost btn-sm" onclick="PrismApp.deleteDoc(${d.id})">Удалить</button>
        </div>
      </div>`).join('') || '<p class="muted">Пока пусто</p>';

    const pager = document.getElementById('docs-pager');
    const label = document.getElementById('docs-page-label');
    const meta = document.getElementById('docs-meta');
    const prev = document.getElementById('docs-prev');
    const next = document.getElementById('docs-next');
    if (meta) meta.textContent = data.total ? `всего ${data.total}` : '';
    if (pager) {
      pager.classList.toggle('hidden', pages <= 1);
      if (label) label.textContent = `стр. ${this._docsPage} из ${pages}`;
      if (prev) prev.disabled = this._docsPage <= 1;
      if (next) next.disabled = this._docsPage >= pages;
    }
    return pages;
  },

  async deleteDoc(id) {
    const ok = await this.confirm({
      title: 'Удалить документ?',
      body: 'Документ и все его фрагменты будут удалены из базы знаний.',
      confirmLabel: 'Удалить',
      danger: true,
    });
    if (!ok) return;
    const res = await fetch(`/api/documents/${id}`, { method: 'DELETE' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      alert(data.detail || 'Не удалось удалить');
      return;
    }
    const pages = await this.loadDocuments(this._docsPage, this._docsPageSize);
    if (this._docsPage > pages) await this.loadDocuments(pages, this._docsPageSize);
  },

  async clearKnowledgeBase() {
    const ok = await this.confirm({
      title: 'Очистить базу знаний?',
      body: 'Будут удалены все документы, фрагменты и загруженные файлы. Это действие нельзя отменить.',
      confirmLabel: 'Очистить базу',
      danger: true,
    });
    if (!ok) return;
    const res = await fetch('/api/documents', { method: 'DELETE' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      alert(typeof data.detail === 'string' ? data.detail : 'Не удалось очистить базу');
      return;
    }
    const status = document.getElementById('job-status');
    if (status) {
      status.textContent = `База очищена · документов ${data.deleted_documents ?? 0} · фрагментов ${data.deleted_chunks ?? 0}`;
    }
    this._docsPage = 1;
    await this.loadDocuments(1, this._docsPageSize);
  },

  async clearJournal() {
    const ok = await this.confirm({
      title: 'Очистить журнал?',
      body: 'Будут удалены все запуски агента и записи поиска/RAG. Это действие нельзя отменить.',
      confirmLabel: 'Очистить журнал',
      danger: true,
    });
    if (!ok) return;
    const res = await fetch('/api/pipeline-logs', { method: 'DELETE' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      alert(typeof data.detail === 'string' ? data.detail : 'Не удалось очистить журнал');
      return false;
    }
    return data;
  },

  /** Promise-based confirm modal. Resolves true/false. */
  confirm({ title, body, confirmLabel = 'Подтвердить', cancelLabel = 'Отмена', danger = false } = {}) {
    return new Promise((resolve) => {
      let root = document.getElementById('prism-modal');
      if (!root) {
        root = document.createElement('div');
        root.id = 'prism-modal';
        root.className = 'modal-root hidden';
        root.innerHTML = `
          <div class="modal-backdrop" data-modal-cancel></div>
          <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="prism-modal-title">
            <h2 id="prism-modal-title" class="modal-title"></h2>
            <p class="modal-body"></p>
            <div class="modal-actions">
              <button type="button" class="btn btn-ghost" data-modal-cancel></button>
              <button type="button" class="btn" data-modal-confirm></button>
            </div>
          </div>`;
        document.body.appendChild(root);
      }

      const titleEl = root.querySelector('.modal-title');
      const bodyEl = root.querySelector('.modal-body');
      const cancelBtn = root.querySelector('[data-modal-cancel].btn');
      const confirmBtn = root.querySelector('[data-modal-confirm]');
      titleEl.textContent = title || 'Подтверждение';
      bodyEl.textContent = body || '';
      cancelBtn.textContent = cancelLabel;
      confirmBtn.textContent = confirmLabel;
      confirmBtn.className = danger ? 'btn btn-danger' : 'btn';

      const close = (value) => {
        root.classList.add('hidden');
        document.removeEventListener('keydown', onKey);
        root.removeEventListener('click', onClick);
        resolve(value);
      };
      const onKey = (e) => {
        if (e.key === 'Escape') close(false);
      };
      const onClick = (e) => {
        if (e.target.closest('[data-modal-confirm]')) close(true);
        else if (e.target.closest('[data-modal-cancel]')) close(false);
      };

      root.addEventListener('click', onClick);
      document.addEventListener('keydown', onKey);
      root.classList.remove('hidden');
      confirmBtn.focus();
    });
  },

  async saveSettings(form) {
    const fd = new FormData(form);
    const body = Object.fromEntries(fd.entries());
    body.chunk_size = Number(body.chunk_size);
    body.chunk_overlap = Number(body.chunk_overlap);
    body.top_k = Number(body.top_k);
    body.rerank_candidates = Number(body.rerank_candidates || 20);
    body.hnsw_ef_search = Number(body.hnsw_ef_search || 40);
    body.rerank_enabled = !!form.querySelector('#rerank_enabled')?.checked;
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    const msg = document.getElementById('settings-msg');
    if (!msg) return;
    if (!res.ok) {
      msg.textContent = data.detail || 'Ошибка';
      return;
    }
    const emb = body.embedding_model || '';
    msg.textContent = (emb === 'hash' || emb === 'hash-embedder')
      ? 'Сохранено. Hash активен — для нормального поиска верните MiniLM и переиндексируйте базу.'
      : (body.rerank_enabled
        ? 'Сохранено. Реранк включён для чата/поиска (оценка по-прежнему из YAML).'
        : 'Сохранено');
  },
};
