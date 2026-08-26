let studyId = null;

const el = id => document.getElementById(id);

// Status indicator updater with visual state flags
const status = (message, state = 'default') => {
  const statusEl = el('status');
  const container = statusEl.closest('.status-container');
  if (statusEl) statusEl.textContent = message;
  
  if (container) {
    container.classList.remove('active', 'complete', 'error');
    if (state === 'active') container.classList.add('active');
    else if (state === 'complete') container.classList.add('complete');
    else if (state === 'error') container.classList.add('error');
  }
};

// File drag & drop + selection handling
const fileInput = el('file');
const dropZone = el('drop-zone');
const promptView = el('drop-zone-prompt');
const fileInfoCard = el('file-info-card');
const selectedFileName = el('selected-file-name');
const selectedFileSize = el('selected-file-size');
const btnChangeFile = el('btn-change-file');

function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function handleFileSelection(file) {
  if (!file) return;
  selectedFileName.textContent = file.name;
  selectedFileSize.textContent = formatBytes(file.size);
  promptView.style.display = 'none';
  fileInfoCard.style.display = 'flex';
  el('analyze').disabled = true;
  status(`Scan loaded: "${file.name}". Click "Upload Scan" to prepare analysis.`, 'default');
}

if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  });
}

if (dropZone) {
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files[0]) {
      fileInput.files = files;
      handleFileSelection(files[0]);
    }
  });
}

if (btnChangeFile) {
  btnChangeFile.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.value = '';
    fileInfoCard.style.display = 'none';
    promptView.style.display = 'flex';
    el('analyze').disabled = true;
    studyId = null;
    status('Select an MRI scan above to begin diagnostic pipeline.', 'default');
  });
}

// Upload scan handler
el('upload').onclick = async () => {
  const file = fileInput.files[0];
  if (!file) return status('Please select or drop an MRI file first.', 'error');

  const uploadBtn = el('upload');
  const originalText = uploadBtn.innerHTML;
  uploadBtn.disabled = true;
  uploadBtn.innerHTML = `<span>Ingesting...</span>`;

  const data = new FormData();
  data.append('file', file);
  status('Uploading scan & initializing DICOM pipeline…', 'active');

  try {
    const r = await fetch('/api/upload', { method: 'POST', body: data });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail);
    studyId = d.study_id;
    el('analyze').disabled = false;
    status('Upload complete. Ready for neural segmentation analysis.', 'complete');
  } catch (e) {
    status(`Upload failed: ${e.message}`, 'error');
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.innerHTML = originalText;
  }
};

// Analyze MRI handler
el('analyze').onclick = async () => {
  if (!studyId) return status('Please upload a scan first.', 'error');

  const analyzeBtn = el('analyze');
  const originalHtml = analyzeBtn.innerHTML;
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<span>Segmenting MRI...</span>`;

  status('Executing deep neural spatial segmentation & lesion boundary detection…', 'active');
  
  try {
    const r = await fetch(`/api/analysis/${studyId}`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) throw Error(d.detail);
    show(d);
    el('scan-another').hidden = false;
    status(`Analysis completed successfully. Identified ${d.tumor_count} lesion region(s).`, 'complete');
  } catch (e) {
    status(`Analysis failed: ${e.message}`, 'error');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = originalHtml;
  }
};

function show(d) {
  const resultsSec = el('results');
  resultsSec.hidden = false;
  resultsSec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  el('count').textContent = d.tumor_count;
  el('model').textContent = `${d.model.name} ${d.model.version}`;
  el('annotated').src = d.annotated_image_url;
  
  const bar = el('metric-bar-lesions');
  if (bar) {
    const widthPct = Math.min(100, Math.max(15, d.tumor_count * 30));
    bar.style.width = `${widthPct}%`;
  }

  el('warning').textContent = d.warnings && d.warnings.length ? d.warnings.join(' ') : '';

  const tumorRows = d.tumors.map(t => `
    <tr>
      <td><strong>${t.tumor_id}</strong></td>
      <td>${t.area_pixels.toLocaleString()}</td>
      <td>${t.width_pixels} × ${t.height_pixels}</td>
      <td>${t.max_diameter_pixels.toFixed(1)} px</td>
      <td>(${t.centroid.map(v => v.toFixed(1)).join(', ')})</td>
    </tr>
  `).join('') || '<tr><td colspan="5" style="text-align:center;color:#94a3b8;">No abnormal hyperintense regions identified.</td></tr>';
  el('tumors').querySelector('tbody').innerHTML = tumorRows;

  const pairRows = d.pairwise_analysis.map(p => `
    <tr>
      <td><strong>${p.tumor_a} ↔ ${p.tumor_b}</strong></td>
      <td>${p.centroid_distance_pixels.toFixed(1)} px</td>
      <td>${p.boundary_distance_pixels.toFixed(1)} px</td>
      <td><span class="badge-pill">${p.relative_position}</span></td>
    </tr>
  `).join('') || '<tr><td colspan="4" style="text-align:center;color:#94a3b8;">Single lesion focus — pairwise analysis not required.</td></tr>';
  el('pairs').querySelector('tbody').innerHTML = pairRows;
}

// PDF Summary Report
el('pdf').onclick = async () => {
  if (!studyId) return;
  status('Generating radiology summary PDF…', 'active');
  try {
    const r = await fetch(`/api/reports/${studyId}/pdf`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) return status(`Report failed: ${d.detail}`, 'error');
    window.open(d.download_url, '_blank');
    status('Radiology summary PDF ready for download.', 'complete');
  } catch (e) {
    status(`Report error: ${e.message}`, 'error');
  }
};

// Doctor's Report PDF
el('doctor-report').onclick = async () => {
  if (!studyId) return;
  status("Generating clinical Doctor's Comprehensive Report PDF…", 'active');
  try {
    const r = await fetch(`/api/doctor-report/${studyId}`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) return status(`Doctor's report failed: ${d.detail}`, 'error');
    window.open(d.download_url, '_blank');
    status(`Doctor's Comprehensive Report PDF opened.`, 'complete');
  } catch (e) {
    status(`Doctor's report error: ${e.message}`, 'error');
  }
};

// Scan Another — full UI reset
el('scan-another').onclick = () => {
  // Reset file input & dropzone
  fileInput.value = '';
  promptView.style.display = 'flex';
  fileInfoCard.style.display = 'none';

  // Reset buttons
  el('analyze').disabled = true;
  el('scan-another').hidden = true;

  // Hide results
  el('results').hidden = true;

  // Clear state
  studyId = null;

  // Reset status
  status('Select or drop an MRI image to begin.', 'default');

  // Scroll back to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
};
