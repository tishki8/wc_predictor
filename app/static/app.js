/**
 * WC Predictor — Frontend Logic
 * ══════════════════════════════
 * Handles:
 *  - Searchable team dropdown
 *  - Stage pill selector
 *  - Neutral toggle label
 *  - Prediction API call + result rendering
 *    • Group stage: simple 3-segment prob bar
 *    • Knockout: layered funnel (90min → ET → Pens → Advances)
 *  - Backtest stats loading
 *  - Live fixtures table population
 */

'use strict';

/* ── State ─────────────────────────────────────────────────────── */
let allTeams    = [];
let selectedStage = 'group';

/* ── Stage → importance (mirrors backend, for info only) ───────── */
const STAGE_IMPORTANCE = {
  group: 25, r32: 50, r16: 50, qf: 60, sf: 60, final: 60,
};

const STAGE_LABELS = {
  group: 'Group Stage', r32: 'Round of 32', r16: 'Round of 16',
  qf: 'Quarter-Final', sf: 'Semi-Final', final: 'Final',
};

/* ══════════════════════════════════════════════════════════════════
   SEARCHABLE TEAM DROPDOWN
   ══════════════════════════════════════════════════════════════════ */

function buildTeamDropdown(prefix, teams) {
  const wrapper    = document.getElementById(`${prefix}-wrapper`);
  const displayBtn = document.getElementById(`${prefix}-display`);
  const displayTxt = document.getElementById(`${prefix}-display-text`);
  const searchDiv  = document.getElementById(`${prefix}-search`);
  const searchInput= document.getElementById(`${prefix}-search-input`);
  const searchList = document.getElementById(`${prefix}-search-list`);
  const hiddenVal  = document.getElementById(`${prefix}-team-val`);

  function renderList(filter = '') {
    const q = filter.toLowerCase();
    const filtered = teams.filter(t => t.toLowerCase().includes(q));
    searchList.innerHTML = '';
    if (filtered.length === 0) {
      searchList.innerHTML = '<div class="team-search-item" style="color:var(--white-40);cursor:default;">No results</div>';
      return;
    }
    filtered.forEach(team => {
      const item = document.createElement('div');
      item.className = 'team-search-item';
      item.textContent = team;
      item.addEventListener('mousedown', (e) => {
        e.preventDefault(); // prevent blur closing before click
        selectTeam(team);
      });
      searchList.appendChild(item);
    });
  }

  function selectTeam(team) {
    hiddenVal.value = team;
    displayTxt.textContent = team;
    closeDropdown();
  }

  function openDropdown() {
    wrapper.classList.add('open');
    displayBtn.classList.add('open');
    displayBtn.setAttribute('aria-expanded', 'true');
    renderList('');
    searchInput.value = '';
    searchInput.focus();
  }

  function closeDropdown() {
    wrapper.classList.remove('open');
    displayBtn.classList.remove('open');
    displayBtn.setAttribute('aria-expanded', 'false');
  }

  displayBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (wrapper.classList.contains('open')) {
      closeDropdown();
    } else {
      // Close any other open dropdowns
      document.querySelectorAll('.select-wrapper.open').forEach(w => {
        w.classList.remove('open');
        w.querySelector('.team-display-btn')?.classList.remove('open');
      });
      openDropdown();
    }
  });

  searchInput.addEventListener('input', () => renderList(searchInput.value));

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!wrapper.contains(e.target)) closeDropdown();
  });

  // Close on Escape
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDropdown();
  });
}

/* ══════════════════════════════════════════════════════════════════
   STAGE PILLS
   ══════════════════════════════════════════════════════════════════ */

document.querySelectorAll('.stage-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    document.querySelectorAll('.stage-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    selectedStage = pill.dataset.stage;
  });
});

/* ── Neutral toggle label ─────────────────────────────────────── */
const neutralToggle = document.getElementById('neutral-toggle');
const neutralLabel  = document.getElementById('neutral-label');
neutralToggle.addEventListener('change', () => {
  neutralLabel.textContent = neutralToggle.checked ? 'On (Tournament)' : 'Off (Home Venue)';
});

/* ══════════════════════════════════════════════════════════════════
   PROBABILITY BAR BUILDER
   ══════════════════════════════════════════════════════════════════ */

/**
 * Build a 3-segment probability bar HTML string.
 * @param {number} ph - home probability (0-1)
 * @param {number} pd - draw probability (0-1)
 * @param {number} pa - away probability (0-1)
 * @param {string} homeLabel
 * @param {string} awayLabel
 * @param {string} stepLabel - e.g. "90 Minutes"
 * @param {number|null} stepNum - step number icon (null to hide)
 * @param {boolean} small - use smaller bar height
 */
function buildProbBar(ph, pd, pa, homeLabel, awayLabel, stepLabel, stepNum = null, small = false) {
  const fmtPct = v => `${(v * 100).toFixed(1)}%`;
  const fmtPctShort = v => `${Math.round(v * 100)}%`;
  const smClass = small ? 'prob-bar-sm' : '';

  const stepNumHtml = stepNum !== null
    ? `<span class="step-num">${stepNum}</span>` : '';

  // Only show segment text if wide enough (>10%)
  const homeTxt = ph > 0.12 ? `<span class="prob-val">${fmtPctShort(ph)}</span><span class="prob-tiny">${homeLabel.split(' ')[0]}</span>` : '';
  const drawTxt = pd > 0.10 ? `<span class="prob-val">${fmtPctShort(pd)}</span><span class="prob-tiny">Draw</span>` : '';
  const awayTxt = pa > 0.12 ? `<span class="prob-val">${fmtPctShort(pa)}</span><span class="prob-tiny">${awayLabel.split(' ')[0]}</span>` : '';

  return `
    <div class="prob-bar-wrap">
      <div class="prob-bar-label">
        ${stepNumHtml}
        ${stepLabel}
      </div>
      <div class="prob-bar ${smClass}">
        <div class="prob-seg prob-seg-home" style="width:${ph*100}%">${homeTxt}</div>
        <div class="prob-seg prob-seg-draw" style="width:${pd*100}%">${drawTxt}</div>
        <div class="prob-seg prob-seg-away" style="width:${pa*100}%">${awayTxt}</div>
      </div>
      <div class="prob-bar-ticks">
        <span class="tick-label">${homeLabel} ${fmtPct(ph)}</span>
        <span class="tick-label">Draw ${fmtPct(pd)}</span>
        <span class="tick-label">${awayLabel} ${fmtPct(pa)}</span>
      </div>
    </div>`;
}

/* ══════════════════════════════════════════════════════════════════
   RESULT RENDERING
   ══════════════════════════════════════════════════════════════════ */

function renderResult(data, home, away) {
  const resultSection = document.getElementById('result-section');
  const resultBody    = document.getElementById('result-body');

  document.getElementById('res-home-name').textContent = home;
  document.getElementById('res-away-name').textContent = away;
  document.getElementById('res-stage-badge').textContent = STAGE_LABELS[data.stage] || data.stage;

  const r = data.result;

  if (data.stage === 'group') {
    // ── Group: simple bar ─────────────────────────────────────
    const xgHtml = `xG ${r.xG_h.toFixed(2)} — ${r.xG_a.toFixed(2)}`;
    document.getElementById('res-home-xg').textContent = `xG ${r.xG_h.toFixed(2)}`;
    document.getElementById('res-away-xg').textContent = `xG ${r.xG_a.toFixed(2)}`;

    const mlHtml = `
      <div style="text-align:center; margin-bottom:20px;">
        <span style="font-size:12px; color:var(--white-40); letter-spacing:0.06em; text-transform:uppercase; font-weight:600;">Most Likely Score</span>
        <div style="font-family:var(--font-num); font-size:36px; font-weight:900; letter-spacing:0.02em; margin-top:4px;">${r.ml_score}</div>
        <div style="font-size:11px; color:var(--white-40);">p = ${(r.ml_prob * 100).toFixed(1)}%</div>
      </div>`;

    resultBody.innerHTML = mlHtml + buildProbBar(r.p_h, r.p_d, r.p_a, home, away, 'Full Time Probabilities', null, false);

  } else {
    // ── Knockout: funnel ──────────────────────────────────────
    document.getElementById('res-home-xg').textContent = `xG ${r.xG_h.toFixed(2)}`;
    document.getElementById('res-away-xg').textContent = `xG ${r.xG_a.toFixed(2)}`;

    const ninety  = r.ninety_min;
    const et      = r.extra_time;
    const pens    = r.penalties;
    const adv     = r.advances;

    // Step 2 (ET) is only "shown" if the draw probability from 90 min is non-trivial
    const etIsRelevant = ninety.draw > 0.08;
    // Step 3 (Pens) only meaningful if ET can still draw
    const pensIsRelevant = etIsRelevant && et.draw > 0.05;

    // Penalty bar (50/50-ish, always literally 50/50 in the model)
    const fmtPensBar = () => {
      const ph = pens.home, pa = pens.away;
      const fmtPct = v => `${Math.round(v * 100)}%`;
      return `
        <div class="prob-bar-wrap">
          <div class="prob-bar-label">
            <span class="step-num">3</span>
            Penalty Shootout
          </div>
          <div class="prob-bar">
            <div class="prob-seg prob-seg-pens-home" style="width:${ph*100}%">
              ${ph > 0.1 ? `<span class="prob-val">${fmtPct(ph)}</span><span class="prob-tiny">${home.split(' ')[0]}</span>` : ''}
            </div>
            <div class="prob-seg prob-seg-pens-away" style="width:${pa*100}%">
              ${pa > 0.1 ? `<span class="prob-val">${fmtPct(pa)}</span><span class="prob-tiny">${away.split(' ')[0]}</span>` : ''}
            </div>
          </div>
          <div class="prob-bar-ticks">
            <span class="tick-label">${home} ${(ph*100).toFixed(0)}%</span>
            <span class="tick-label">${away} ${(pa*100).toFixed(0)}%</span>
          </div>
        </div>`;
    };

    // Advances block
    const homeLeads = adv.home >= adv.away;
    const advHtml = `
      <div class="advances-block">
        <div class="advances-label">⚡ Advances to Next Round</div>
        <div class="advances-row">
          <div class="advances-team">
            <div class="advances-team-name">${home}</div>
            <div class="advances-prob ${homeLeads ? 'leader' : 'trailer'}">
              ${(adv.home * 100).toFixed(1)}%
            </div>
          </div>
          <div class="advances-divider">vs</div>
          <div class="advances-team">
            <div class="advances-team-name">${away}</div>
            <div class="advances-prob ${!homeLeads ? 'leader' : 'trailer'}">
              ${(adv.away * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>`;

    // Build nested funnel HTML
    let funnelHtml = `<div class="funnel-wrap">`;

    // Step 1: 90 minutes
    funnelHtml += `<div class="funnel-step">
      ${buildProbBar(ninety.home, ninety.draw, ninety.away, home, away, '90 Minutes', 1)}
    </div>`;

    if (etIsRelevant) {
      // Connector + ET nested block
      funnelHtml += `
      <div class="funnel-connector">
        <div class="funnel-connector-line"></div>
        <div class="funnel-connector-label">If drawn after 90 min (${(ninety.draw * 100).toFixed(0)}% chance) →</div>
      </div>
      <div class="funnel-nested">
        <div class="funnel-step">
          ${buildProbBar(et.home, et.draw, et.away, home, away, 'Extra Time', 2, true)}
        </div>`;

      if (pensIsRelevant) {
        funnelHtml += `
        <div class="funnel-connector">
          <div class="funnel-connector-line"></div>
          <div class="funnel-connector-label">If still drawn after ET (${(et.draw * 100).toFixed(0)}% chance) →</div>
        </div>
        <div class="funnel-nested">
          ${fmtPensBar()}
        </div>`;
      }

      funnelHtml += `</div>`; // close funnel-nested
    }

    funnelHtml += `</div>`; // close funnel-wrap

    resultBody.innerHTML = funnelHtml + advHtml;
  }

  // Animate bars in after layout
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.prob-seg').forEach(seg => {
        const w = seg.style.width;
        seg.style.width = '0%';
        requestAnimationFrame(() => { seg.style.width = w; });
      });
    });
  });

  resultSection.classList.add('visible');
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ══════════════════════════════════════════════════════════════════
   PREDICTION API CALL
   ══════════════════════════════════════════════════════════════════ */

async function runPrediction() {
  const home    = document.getElementById('home-team-val').value.trim();
  const away    = document.getElementById('away-team-val').value.trim();
  const neutral = document.getElementById('neutral-toggle').checked;
  const stage   = selectedStage;

  const errorEl   = document.getElementById('error-msg');
  const spinner   = document.getElementById('predict-spinner');
  const btn       = document.getElementById('predict-btn');

  errorEl.style.display = 'none';
  errorEl.textContent   = '';

  if (!home || !away) {
    showError('Please select both teams before predicting.'); return;
  }
  if (home === away) {
    showError('Please select two different teams.'); return;
  }

  btn.disabled = true;
  spinner.style.display = 'block';

  try {
    const resp = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ home, away, stage, neutral }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      showError(data.error || `Server error: ${resp.status}`);
      return;
    }

    renderResult(data, home, away);

  } catch (err) {
    showError(`Network error: ${err.message}`);
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
  }
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

/* ══════════════════════════════════════════════════════════════════
   BACKTEST PANEL
   ══════════════════════════════════════════════════════════════════ */

async function loadBacktest() {
  try {
    const resp = await fetch('/api/backtest');
    const d    = await resp.json();

    document.getElementById('bt-accuracy').textContent =
      `${(d.accuracy * 100).toFixed(1)}%`;
    document.getElementById('bt-brier').textContent =
      d.brier_score.toFixed(3);
    document.getElementById('bt-logloss').textContent =
      d.log_loss.toFixed(3);
    document.getElementById('backtest-subtitle').textContent =
      `${d.n_matches} matches · ${d.tournament} · Honest edge over random baseline`;

  } catch (e) {
    console.warn('Backtest load failed', e);
  }
}

/* ══════════════════════════════════════════════════════════════════
   LIVE FIXTURES TABLE
   ══════════════════════════════════════════════════════════════════ */

function fmtPct(v, decimals = 1) {
  if (v === null || v === undefined) return '—';
  return `${(v * 100).toFixed(decimals)}%`;
}

function fmtDate(s) {
  if (!s) return '—';
  const d = new Date(s);
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
}

function stagePill(s) {
  if (!s) return '';
  const labels = { group: 'Group', r32: 'R32', r16: 'R16', qf: 'QF', sf: 'SF', final: 'Final' };
  return `<span class="stage-tag">${labels[s.toLowerCase()] || s.toUpperCase()}</span>`;
}

async function loadFixtures() {
  const tbody = document.getElementById('fixtures-body');
  try {
    const resp = await fetch('/api/live-fixtures');
    const rows = await resp.json();

    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:24px;color:var(--white-40);font-style:italic;">No fixtures yet.</td></tr>';
      return;
    }

    tbody.innerHTML = rows.map(row => {
      const isKnockout = row.stage && row.stage.toLowerCase() !== 'group';
      const advHtml = isKnockout
        ? `<td class="prob-num advances" style="text-align:right;">${fmtPct(row.p_home_advances)} / ${fmtPct(row.p_away_advances)}</td>`
        : `<td class="prob-num" style="text-align:right;color:var(--white-15);font-style:italic;font-size:11px;">—</td>`;

      return `<tr>
        <td>
          <div class="fixture-teams">
            <span class="fixture-home">${row.home_team || '—'}</span>
            <span class="fixture-vs">vs</span>
            <span class="fixture-away">${row.away_team || '—'}</span>
          </div>
        </td>
        <td class="fixture-date">${fmtDate(row.match_date)}</td>
        <td>${stagePill(row.stage)}</td>
        <td class="prob-num home" style="text-align:right;">${fmtPct(row.p_home)}</td>
        <td class="prob-num draw" style="text-align:right;">${fmtPct(row.p_draw)}</td>
        <td class="prob-num away" style="text-align:right;">${fmtPct(row.p_away)}</td>
        ${advHtml}
        <td class="result-indicator">Pending</td>
      </tr>`;
    }).join('');

  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:24px;color:#fca5a5;">Failed to load fixtures.</td></tr>`;
    console.error('Fixtures load error', e);
  }
}

/* ══════════════════════════════════════════════════════════════════
   INIT
   ══════════════════════════════════════════════════════════════════ */

async function init() {
  try {
    const resp = await fetch('/api/teams');
    allTeams   = await resp.json();
  } catch (e) {
    console.error('Failed to load teams', e);
    allTeams = [];
  }

  buildTeamDropdown('home', allTeams);
  buildTeamDropdown('away', allTeams);

  // Allow Enter key on predict button
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.closest('#selector') && !e.target.classList.contains('team-search-input')) {
      runPrediction();
    }
  });

  await Promise.all([loadBacktest(), loadFixtures()]);
}

document.addEventListener('DOMContentLoaded', init);
