let studyId = null;

const el = id => document.getElementById(id);
const status = message => { el('status').textContent = message; };

el('upload').onclick = async () => {
  const file = el('file').files[0];
  if (!file) return status('Please select an MRI file.');

  const data = new FormData();
  data.append('file', file);
  status('Uploading scan…');

  try {
    const r = await fetch('/api/upload', { method: 'POST', body: data });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail);
    studyId = d.study_id;
    el('analyze').disabled = false;
    status('Upload complete. Ready to analyze.');
  } catch (e) {
    status(`Upload failed: ${e.message}`);
  }
};

el('analyze').onclick = async () => {
  status('Running tumor segmentation and spatial analysis…');
  try {
    const r = await fetch(`/api/analysis/${studyId}`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail);
    show(d);
    status('Analysis complete.');
  } catch (e) {
    status(`Analysis failed: ${e.message}`);
  }
};

function show(d) {
  el('results').hidden = false;
  el('count').textContent = d.tumor_count;
  el('model').textContent = `${d.model.name} ${d.model.version}`;
  el('annotated').src = d.annotated_image_url;
  el('warning').textContent = d.warnings.join(' ');

  const tumorRows = d.tumors.map(t => `
    <tr>
      <td><strong>${t.tumor_id}</strong></td>
      <td>${t.area_pixels.toLocaleString()}</td>
      <td>${t.width_pixels} × ${t.height_pixels}</td>
      <td>${t.max_diameter_pixels.toFixed(1)}</td>
      <td>${t.centroid.map(v => v.toFixed(1)).join(', ')}</td>
    </tr>
  `).join('') || '<tr><td colspan="5">No abnormal regions detected.</td></tr>';
  el('tumors').querySelector('tbody').innerHTML = tumorRows;

  const pairRows = d.pairwise_analysis.map(p => `
    <tr>
      <td><strong>${p.tumor_a}–${p.tumor_b}</strong></td>
      <td>${p.centroid_distance_pixels.toFixed(1)}</td>
      <td>${p.boundary_distance_pixels.toFixed(1)}</td>
      <td>${p.relative_position}</td>
    </tr>
  `).join('') || '<tr><td colspan="4">No pairs available.</td></tr>';
  el('pairs').querySelector('tbody').innerHTML = pairRows;
}

el('pdf').onclick = async () => {
  if (!studyId) return;
  status('Generating PDF report…');
  try {
    const r = await fetch(`/api/reports/${studyId}/pdf`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) return status(`Report failed: ${d.detail}`);
    window.open(d.download_url, '_blank');
    status('PDF report ready.');
  } catch (e) {
    status(`Report error: ${e.message}`);
  }
};
