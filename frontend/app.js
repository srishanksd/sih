const canvas = document.getElementById('map'), ctx = canvas.getContext('2d'), slider = document.getElementById('time');
const missing = v => v === null || v === undefined || Number.isNaN(v) ? '-' : v;

function draw(field, storms) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!Array.isArray(field) || !field.length) {
        ctx.fillStyle = '#888'; ctx.font = '14px Poppins';
        ctx.fillText('No radar field available', 24, 30); return;
    }
    const rows = field.length, cols = field[0].length;
    const cw = canvas.width / cols, ch = canvas.height / rows;
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
        const v = Math.max(0, Math.min(1, Number(field[y][x]) || 0));
        const r = Math.round(245 - 170 * v), g = Math.round(247 - 130 * v), b = Math.round(250 - 40 * v);
        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.fillRect(x * cw, y * ch, cw + 1, ch + 1);
    }
    (storms || []).forEach(s => {
        const x = s.x / 96 * canvas.width, y = s.y / 96 * canvas.height;
        const r = Math.max(7, Math.sqrt(s.area || 1) * 1.8);
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.strokeStyle = '#333'; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = '#222'; ctx.font = '600 11px Poppins';
        ctx.fillText(s.id || 'cell', x + r + 5, y + 4);
    });
}
function row(name, value) { return '<div class="row"><span class="name">' + name + '</span><span class="value">' + value + '</span></div>'; }
function stateRow(name, value, suffix='') {
    return '<div class="state-card"><span>' + name + '</span><strong>' + value + suffix + '</strong></div>';
}

async function load(t) {
    try {
        const d = await fetch('/api/state?t=' + t).then(r => r.json());
        const storms = Array.isArray(d.objects) ? d.objects : [];
        document.getElementById('tv').textContent = t == 0 ? 'Observed' : 'T+' + t + ' min';
        document.getElementById('rangeval').textContent = t + ' min';
        document.getElementById('cells').textContent = storms.length || '-';
        document.getElementById('lead').textContent = t == 0 ? 'Observed' : t + ' min';
        document.getElementById('quality').textContent = d.quality != null ? Math.round(d.quality * 100) + '%' : '-';
        document.getElementById('datastatus').textContent = d.mode === 'prototype_replay' ? 'Replay' : '-';
        document.getElementById('mode').textContent = d.mode === 'prototype_replay' ? 'DWR + INSAT-3DR · ' + (t == 0 ? 'observed' : 'model forecast') : 'No data';
        draw(d.field, storms);

        const cs = d.convective_state || {};
        document.getElementById('state-badge').textContent = cs.cells > 0 ? cs.cells + ' active cell' + (cs.cells > 1 ? 's' : '') : 'Field state';
        document.getElementById('convective-state').innerHTML =
            stateRow('Cells', cs.cells ?? '-') +
            stateRow('Mean intensity', cs.mean_intensity == null ? '-' : cs.mean_intensity.toFixed(2)) +
            stateRow('Max intensity', cs.max_intensity == null ? '-' : cs.max_intensity.toFixed(2)) +
            stateRow('Growth proxy', cs.growth_proxy == null ? '-' : cs.growth_proxy.toFixed(2)) +
            stateRow('Motion magnitude', cs.mean_motion == null ? '-' : cs.mean_motion.toFixed(2)) +
            stateRow('Convective area', cs.convective_fraction == null ? '-' : Math.round(cs.convective_fraction * 100), '%') +
            stateRow('Strong-cell area', cs.strong_fraction == null ? '-' : Math.round(cs.strong_fraction * 100), '%');

        const sensors = d.sensors || {};
        document.getElementById('sensors').innerHTML = ['DWR','INSAT','ILDN','AWS','GFS'].map(k => {
            const v = sensors[k]; return '<div class="row"><span class="name">' + k + '</span><span class="value">' + (v == null ? '-' : Math.round(v * 100) + '%') + '</span></div>';
        }).join('');
        const hz = d.hazards || {};
        document.getElementById('hazards').innerHTML = ['thunderstorm','lightning','hail','cloudburst'].map(k =>
            row(k[0].toUpperCase() + k.slice(1), hz[k] == null ? '-' : Math.round(hz[k] * 100) + '%')).join('');
        document.getElementById('storms').innerHTML =
            '<div class="table-head"><span>ID</span><span>Intensity</span><span>Motion</span><span>Quality</span></div>' +
            (storms.length ? storms.map(s =>
                '<div class="storm-row"><span>' + missing(s.id) + '</span><span>' +
                (s.intensity == null ? '-' : s.intensity.toFixed(2)) + '</span><span>' +
                (s.dx == null || s.dy == null ? '-' : s.dx.toFixed(1) + ', ' + s.dy.toFixed(1)) + '</span><span>' +
                (s.state?.quality == null ? '-' : Math.round(s.state.quality * 100) + '%') + '</span></div>').join('') :
            '<div class="empty">No storm cells detected at this lead time.</div>');
    } catch (e) {
        document.getElementById('datastatus').textContent = '-';
        document.getElementById('mode').textContent = 'Prototype unavailable';
        draw(null, []);
    }
}
slider.oninput = () => load(slider.value);
let playing = false;
document.getElementById('play').onclick = () => { playing = !playing; document.getElementById('play').textContent = playing ? 'Pause' : 'Play'; if (playing) step(); };
function step() {
    if (!playing) return;
    const next = (+slider.value + 15) > 60 ? 0 : (+slider.value + 15);
    slider.value = next; load(next); setTimeout(step, 900);
}
load(0);
