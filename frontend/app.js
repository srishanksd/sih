const canvas = document.getElementById('map'), ctx = canvas.getContext('2d'), slider = document.getElementById('time');
const missing = v => v === null || v === undefined || Number.isNaN(v) ? '-' : v;

function draw(field, storms) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#f1f1f1'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!Array.isArray(field) || !field.length) {
        ctx.fillStyle = '#777'; ctx.font = '600 16px Poppins'; ctx.fillText('No radar field available', 24, 34); return;
    }
    const rows = field.length, cols = field[0].length, cw = canvas.width / cols, ch = canvas.height / rows;
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
        const v = Math.max(0, Math.min(1, Number(field[y][x]) || 0));
        const dbz = v * 80 - 10;
        let shade = 245 - Math.round(Math.max(0, Math.min(75, dbz + 10)) / 75 * 220);
        if (dbz < 10) shade = 245;
        ctx.fillStyle = `rgb(${shade},${shade},${shade})`; ctx.fillRect(x * cw, y * ch, cw + 1, ch + 1);
    }
    ctx.strokeStyle = 'rgba(70,70,70,.25)'; ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) { const gx = canvas.width * i / 5, gy = canvas.height * i / 5; ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, canvas.height); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(canvas.width, gy); ctx.stroke(); }
    ctx.strokeStyle = '#222'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(canvas.width / 2 - 9, canvas.height / 2); ctx.lineTo(canvas.width / 2 + 9, canvas.height / 2); ctx.stroke(); ctx.beginPath(); ctx.moveTo(canvas.width / 2, canvas.height / 2 - 9); ctx.lineTo(canvas.width / 2, canvas.height / 2 + 9); ctx.stroke();
    ctx.fillStyle = '#222'; ctx.font = '600 11px Poppins'; ctx.fillText('Cherrapunji DWR', canvas.width / 2 + 12, canvas.height / 2 - 10); ctx.font = '10px Poppins'; ctx.fillStyle = '#555'; ctx.fillText('25.268°N, 91.733°E', canvas.width / 2 + 12, canvas.height / 2 + 5); ctx.fillText('Approx. 600 km radar domain', 14, canvas.height - 14);
    (storms || []).forEach(s => { const x = s.x / 96 * canvas.width, y = s.y / 96 * canvas.height, r = Math.max(7, Math.sqrt(s.area || 1) * 1.8); ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.strokeStyle = '#111'; ctx.lineWidth = 2; ctx.stroke(); ctx.fillStyle = '#111'; ctx.font = '600 11px Poppins'; ctx.fillText(s.id || 'cell', x + r + 5, y + 4); });
}
function row(name, value, cls = '') { return `<div class="row"><span class="name">${name}</span><span class="value ${cls}">${value}</span></div>`; }
function stateRow(name, value, suffix = '') { return `<div class="state-card"><span>${name}</span><strong>${value}${suffix}</strong></div>`; }

async function load(t) {
    try {
        const d = await fetch('/api/state?t=' + t).then(r => r.json()), storms = Array.isArray(d.objects) ? d.objects : [], observed = Number(t) === 0;
        document.getElementById('tv').textContent = observed ? 'Observed' : 'T+' + t + ' min'; document.getElementById('rangeval').textContent = t + ' min'; document.getElementById('cells').textContent = storms.length || '-'; document.getElementById('lead').textContent = observed ? 'Observed' : t + ' min';
        const sensors = d.sensors || {}, details = d.sensor_details || {}; const available = ['DWR', 'INSAT', 'ILDN', 'GFS'].filter(k => Number(sensors[k]) > 0).length;
        document.getElementById('quality').textContent = `${available} / 5`; document.getElementById('datastatus').textContent = d.mode === 'prototype_replay' ? 'Replay ready' : 'No data'; document.getElementById('mode').textContent = observed ? 'DWR observation' : 'DWR + model forecast';
        document.getElementById('map-title').textContent = observed ? 'DWR Radar Reflectivity' : 'Forecast Radar Field'; document.getElementById('map-subtitle').textContent = observed ? 'Observed radar echo · stronger returns indicate stronger precipitation echoes' : 'Model forecast · estimated evolution of the current convective field'; document.getElementById('legend-note').textContent = observed ? 'Approx. reflectivity · dBZ' : 'Forecast field · relative intensity'; draw(d.field, storms);
        const cs = d.convective_state || {}; document.getElementById('state-badge').textContent = cs.cells > 0 ? `${cs.cells} active cell${cs.cells > 1 ? 's' : ''}` : 'No detected cell'; document.getElementById('convective-state').innerHTML = stateRow('Detected cells', cs.cells ?? '-') + stateRow('Mean intensity', cs.mean_intensity == null ? '-' : cs.mean_intensity.toFixed(2)) + stateRow('Max intensity', cs.max_intensity == null ? '-' : cs.max_intensity.toFixed(2)) + stateRow('Growth proxy', cs.growth_proxy == null ? '-' : cs.growth_proxy.toFixed(2)) + stateRow('Motion magnitude', cs.mean_motion == null ? '-' : cs.mean_motion.toFixed(2)) + stateRow('Convective area', cs.convective_fraction == null ? '-' : Math.round(cs.convective_fraction * 100), '%') + stateRow('Strong-cell area', cs.strong_fraction == null ? '-' : Math.round(cs.strong_fraction * 100), '%');
        document.getElementById('sensors').innerHTML = ['DWR', 'INSAT', 'ILDN', 'AWS', 'GFS'].map(k => {
            const det = details[k] || {}, isArchive = k === 'AWS' && det.available; let label = det.label || (Number(sensors[k]) > 0 ? 'Available' : 'Not available');
            let cls = Number(sensors[k]) > 0 ? 'status-ok' : 'status-na'; if (isArchive) { label = 'Historical archive'; cls = 'status-ok'; }
            return row(k, label, cls);
        }).join('');
        const aws = details.AWS; document.querySelector('.panel-footnote').innerHTML = aws?.available ? `<strong>AWS:</strong> ${aws.station || 'Historical station'} · ${aws.date_start || '?'} to ${aws.date_end || '?'} · ${aws.records || 0} records. It is displayed as an archive because it does not overlap this 2026 replay.` : 'A missing source is shown as <strong>Not available</strong>, not as zero weather activity.';
        const hz = d.hazards || {}; document.getElementById('hazards').innerHTML = ['thunderstorm', 'lightning', 'hail', 'cloudburst'].map(k => row(k[0].toUpperCase() + k.slice(1), hz[k] == null ? '-' : Math.round(hz[k] * 100) + '%')).join('');
        document.getElementById('storms').innerHTML = '<div class="table-head"><span>ID</span><span>Intensity</span><span>Motion</span><span>Quality</span></div>' + (storms.length ? storms.map(s => `<div class="storm-row"><span>${missing(s.id)}</span><span>${s.intensity == null ? '-' : s.intensity.toFixed(2)}</span><span>${s.dx == null || s.dy == null ? '-' : s.dx.toFixed(1) + ', ' + s.dy.toFixed(1)}</span><span>${s.state?.quality == null ? '-' : Math.round(s.state.quality * 100) + '%'}</span></div>`).join('') : '<div class="empty">No storm cells detected at this lead time.</div>');
    } catch (e) { document.getElementById('datastatus').textContent = 'Unavailable'; document.getElementById('mode').textContent = 'Prototype unavailable'; draw(null, []); }
}
slider.oninput = () => load(slider.value); let playing = false; document.getElementById('play').onclick = () => { playing = !playing; document.getElementById('play').textContent = playing ? 'Pause' : 'Play'; if (playing) step(); }; function step() { if (!playing) return; const next = (+slider.value + 15) > 60 ? 0 : (+slider.value + 15); slider.value = next; load(next); setTimeout(step, 900); } load(0);
