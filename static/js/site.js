// Interactive bits for the AgentSSL project page. Plain JS, no dependencies
// (highlight.js is optional and only used if it loaded).
(function () {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- 1. Scrollspy for the sticky nav ----------
  function initScrollspy() {
    const links = Array.from(document.querySelectorAll('.site-nav a[href^="#"]'));
    const targets = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    if (!('IntersectionObserver' in window) || !targets.length) return;
    const byId = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
    const visible = new Set();
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id)));
      // Highlight the first visible section in document order.
      const current = targets.find(t => visible.has(t.id));
      links.forEach(a => a.classList.remove('is-active'));
      if (current) byId.get(current.id).classList.add('is-active');
    }, { rootMargin: '-45% 0px -50% 0px' });
    targets.forEach(t => io.observe(t));
  }

  // ---------- 2. Fade sections in on scroll ----------
  function initReveal() {
    const els = document.querySelectorAll('.reveal');
    if (reduceMotion || !('IntersectionObserver' in window)) {
      els.forEach(el => el.classList.add('is-visible'));
      return;
    }
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) { e.target.classList.add('is-visible'); io.unobserve(e.target); }
      });
    }, { threshold: 0.08 });
    els.forEach(el => io.observe(el));
  }

  // ---------- 3. Results table: filters + crosshair hover ----------
  function initResultsTable() {
    const table = document.querySelector('.results-table');
    if (!table) return;
    const rows = Array.from(table.querySelectorAll('tbody tr'));
    const state = { agg: 'all', fb: 'all' };

    function apply() {
      rows.forEach(r => {
        const agg = r.dataset.agg, fb = r.dataset.fb;
        const show = !agg || // baselines and the delta row are always shown
          ((state.agg === 'all' || state.agg === agg) && (state.fb === 'all' || state.fb === fb));
        r.hidden = !show;
      });
    }
    document.querySelectorAll('.table-filter').forEach(group => {
      group.addEventListener('click', e => {
        const btn = e.target.closest('button');
        if (!btn) return;
        group.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
        state[group.dataset.key] = btn.dataset.value;
        apply();
      });
    });

    // Crosshair: highlight the hovered row and column.
    const colCells = idx => table.querySelectorAll(
      `tbody tr td:nth-child(${idx + 1}), thead tr:last-child th:nth-child(${idx + 1})`);
    let lit = [];
    table.addEventListener('mouseover', e => {
      const td = e.target.closest('tbody td');
      lit.forEach(c => c.classList.remove('xhair'));
      lit = [];
      if (!td || td.cellIndex === 0) return;
      lit = Array.from(colCells(td.cellIndex));
      lit.forEach(c => c.classList.add('xhair'));
    });
    table.addEventListener('mouseleave', () => { lit.forEach(c => c.classList.remove('xhair')); lit = []; });
  }

  // ---------- 4. Interactive transfer heatmap ----------
  function initHeatmap() {
    const host = document.getElementById('transfer-heatmap');
    if (!host) return;
    const sources = ['DTD', 'RESISC45', 'KITTI'];
    const targets = ['CLEVR', 'KITTI', 'RESISC45', 'Retino', 'SUN397', 'DTD'];
    const vals = [
      [-20.8, -4.7, -23.3, -7.2, -7.0, 0.0],
      [-16.2, -9.2, 0.0, -14.6, -8.1, -5.9],
      [-21.7, 0.0, -27.8, -9.0, -9.3, -11.6],
    ];
    const VMAX = 28;
    const stops = [[0, [255, 249, 212]], [7, [228, 224, 244]], [15, [180, 167, 215]], [28, [91, 71, 125]]];
    const colour = v => {
      const a = Math.min(Math.abs(v), VMAX);
      for (let i = 0; i < stops.length - 1; i++) {
        const [p0, c0] = stops[i], [p1, c1] = stops[i + 1];
        if (a <= p1) {
          const t = (a - p0) / (p1 - p0);
          return `rgb(${c0.map((c, k) => Math.round(c + (c1[k] - c) * t)).join(',')})`;
        }
      }
    };
    const fmt = v => (v === 0 ? '±0.0' : '−' + Math.abs(v).toFixed(1));

    const grid = document.createElement('div');
    grid.className = 'hm-grid';
    grid.style.gridTemplateColumns = `auto repeat(${targets.length}, minmax(3.6rem, 1fr))`;
    grid.setAttribute('role', 'table');
    grid.setAttribute('aria-label', 'Change in accuracy when a source dataset\'s best program runs on each target dataset');

    const cell = (cls, text) => { const d = document.createElement('div'); d.className = cls; d.textContent = text; return d; };
    grid.appendChild(cell('hm-corner', 'source ↓ / target →'));
    targets.forEach(t => grid.appendChild(cell('hm-col', t)));
    sources.forEach((s, i) => {
      grid.appendChild(cell('hm-row', s));
      targets.forEach((t, j) => {
        const v = vals[i][j];
        const c = cell('hm-cell', fmt(v));
        c.style.background = colour(v);
        if (Math.abs(v) > 18) c.classList.add('dark');
        if (s === t) c.classList.add('self');
        c.tabIndex = 0;
        c.dataset.tip = s === t
          ? `${s} → ${t}: in-domain (reference)`
          : `${s}'s best program on ${t}: ${fmt(v)} points vs. in-domain search`;
        c.dataset.row = i; c.dataset.col = j;
        grid.appendChild(c);
      });
    });
    host.appendChild(grid);

    const tip = document.createElement('div');
    tip.className = 'hm-tip';
    tip.setAttribute('role', 'status');
    tip.textContent = 'Hover or tap a cell to read it.';
    host.appendChild(tip);

    const show = c => {
      grid.querySelectorAll('.hm-cell').forEach(x => x.classList.toggle(
        'dim', x !== c && x.dataset.row !== c.dataset.row && x.dataset.col !== c.dataset.col));
      tip.textContent = c.dataset.tip;
    };
    const reset = () => { grid.querySelectorAll('.hm-cell').forEach(x => x.classList.remove('dim')); };
    grid.addEventListener('mouseover', e => { const c = e.target.closest('.hm-cell'); if (c) show(c); });
    grid.addEventListener('focusin', e => { const c = e.target.closest('.hm-cell'); if (c) show(c); });
    grid.addEventListener('mouseleave', reset);
    grid.addEventListener('focusout', reset);
  }

  // ---------- 5. Click-to-zoom figures ----------
  function initLightbox() {
    const box = document.createElement('div');
    box.className = 'lightbox';
    box.hidden = true;
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-modal', 'true');
    box.innerHTML = '<button class="lightbox-close" aria-label="Close">×</button><img alt="">';
    document.body.appendChild(box);
    const big = box.querySelector('img');
    let opener = null;
    const close = () => { box.hidden = true; document.body.classList.remove('no-scroll'); if (opener) opener.focus(); };

    document.querySelectorAll('.figure-card img').forEach(img => {
      img.classList.add('zoomable');
      img.tabIndex = 0;
      img.setAttribute('role', 'button');
      img.setAttribute('aria-label', 'Enlarge figure: ' + img.alt);
      const open = () => {
        opener = img;
        big.src = img.src; big.alt = img.alt;
        box.hidden = false;
        document.body.classList.add('no-scroll');
        box.querySelector('.lightbox-close').focus();
      };
      img.addEventListener('click', open);
      img.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    });
    box.addEventListener('click', e => { if (e.target !== big) close(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && !box.hidden) close(); });
  }

  // ---------- 6. Copy buttons (BibTeX, code) ----------
  function initCopy() {
    document.querySelectorAll('[data-copy]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const src = document.querySelector(btn.dataset.copy);
        if (!src) return;
        const label = btn.querySelector('.copy-label');
        const old = label.textContent;
        try {
          await navigator.clipboard.writeText(src.textContent);
          label.textContent = 'Copied!';
        } catch (_) {
          // Clipboard API unavailable (e.g. file:// preview): select the text instead.
          const r = document.createRange(); r.selectNodeContents(src);
          const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
          label.textContent = 'Press ⌘C / Ctrl+C';
        }
        setTimeout(() => { label.textContent = old; }, 1800);
      });
    });
  }

  // ---------- 7. Syntax highlighting for the program viewer ----------
  function initCode() {
    const details = document.getElementById('retino-program');
    if (!details) return;
    details.addEventListener('toggle', () => {
      const code = details.querySelector('code');
      if (details.open && window.hljs && !code.dataset.highlighted) window.hljs.highlightElement(code);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initScrollspy();
    initReveal();
    initResultsTable();
    initHeatmap();
    initLightbox();
    initCopy();
    initCode();
  });
})();
