/* ResumeX front-end: weight sliders, upload, progress polling, comparison,
   and the one animated moment - the score bar filling on the candidate page. */

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- weight sliders + live stacked preview ---------- */
function initWeights() {
  const inputs = [...document.querySelectorAll('input[data-weight]')];
  if (!inputs.length) return;
  const total = document.getElementById('weight-total');
  const preview = document.querySelectorAll('#weight-preview i');
  const refresh = () => {
    const vals = inputs.map(i => Number(i.value));
    const sum = vals.reduce((a, b) => a + b, 0);
    inputs.forEach(i => {
      const out = document.getElementById('out_' + i.id);
      if (out) out.textContent = i.value + '%';
    });
    preview.forEach((seg, k) => { seg.style.width = (sum ? vals[k] / sum * 100 : 0) + '%'; });
    if (total) {
      total.querySelector('strong').textContent = sum + '%';
      total.className = 'total ' + (sum === 100 ? 'ok' : 'bad');
    }
  };
  inputs.forEach(i => i.addEventListener('input', refresh));
  refresh();
}

/* ---------- upload ---------- */
function initUpload() {
  const zone = document.getElementById('dropzone');
  const input = document.getElementById('file-input');
  if (!zone || !input) return;
  const list = document.getElementById('filelist');
  const count = document.getElementById('file-count');
  const submit = document.getElementById('start-screening');
  const kb = n => n > 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB';

  const show = () => {
    const files = [...(input.files || [])];
    list.innerHTML = files.map(f =>
      `<div><span>${f.name.replace(/</g, '&lt;')}</span><span class="muted">${kb(f.size)}</span></div>`).join('');
    list.classList.toggle('has', files.length > 0);
    count.textContent = files.length ? `${files.length} resume${files.length > 1 ? 's' : ''} ready to screen`
                                     : 'No files chosen yet';
    submit.disabled = files.length === 0;
  };

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', show);
  ['dragenter', 'dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', e => { input.files = e.dataTransfer.files; show(); });
  submit.closest('form').addEventListener('submit', () => {
    submit.disabled = true; submit.textContent = 'Uploading…';
  });
  show();
}

/* ---------- processing screen ---------- */
function initProgress() {
  const box = document.getElementById('progress-box');
  if (!box) return;
  const job = box.dataset.job;
  const pct = document.getElementById('pct');
  const bar = document.getElementById('pct-bar');
  const note = document.getElementById('proc-note');

  const poll = async () => {
    try {
      const d = await (await fetch(`/api/jobs/${job}/progress`)).json();
      const p = d.percent || 0;
      pct.textContent = p + '%';
      bar.style.width = p + '%';
      document.querySelectorAll('#stage-list li').forEach(li => {
        const done = (d.stages_done || []).includes(li.dataset.stage);
        li.classList.toggle('done', done);
        li.classList.toggle('active', !done && d.stage === li.dataset.stage);
        li.querySelector('.tick').textContent = done ? '✓' : '';
      });
      if (d.total) note.textContent = `${d.done || 0} of ${d.total} resumes read`;
      if (d.state === 'done') { window.location = `/jobs/${job}/results`; return; }
      if (d.state === 'error') {
        note.innerHTML = '<span style="color:var(--weak)">Screening stopped: ' +
          (d.errors || []).join(' ').replace(/</g, '&lt;') + '</span>';
        return;
      }
    } catch (e) { /* network blip: keep polling */ }
    setTimeout(poll, 700);
  };
  poll();
}

/* ---------- comparison ---------- */
function initCompare() {
  const boxes = document.querySelectorAll('input.cmp-box');
  const link = document.getElementById('compare-link');
  if (!boxes.length || !link) return;
  const refresh = () => {
    const ids = [...boxes].filter(b => b.checked).map(b => b.value);
    link.href = `${link.dataset.base}?ids=${ids.join(',')}`;
    link.style.display = ids.length >= 2 ? 'inline-flex' : 'none';
    link.textContent = `Compare ${ids.length}`;
    boxes.forEach(b => { b.disabled = !b.checked && ids.length >= 4; });
  };
  boxes.forEach(b => b.addEventListener('change', refresh));
  refresh();
}

/* ---------- candidate page: the score bar fills to show how points add up ---------- */
function initScoreBar() {
  const segs = document.querySelectorAll('#score-bar i');
  if (!segs.length) return;
  const fill = () => segs.forEach(s => { s.style.width = s.dataset.w + '%'; });
  if (reduceMotion) { fill(); return; }
  requestAnimationFrame(() => requestAnimationFrame(fill));
}

document.addEventListener('DOMContentLoaded', () => {
  initWeights(); initUpload(); initProgress(); initCompare(); initScoreBar();
});
