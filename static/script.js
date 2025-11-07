// Health Check
async function checkHealth() {
  const res = await fetch('/health');
  const data = await res.json();
  document.getElementById('healthResult').textContent = JSON.stringify(data, null, 2);
}

// Register Hardware
async function registerHardware() {
  const type = document.getElementById('hardwareType').value || "unknown";
  const res = await fetch('/hardwares', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({type})
  });
  const data = await res.json();
  document.getElementById('hardwareResult').textContent = JSON.stringify(data, null, 2);
}

// Upload Metadata
document.getElementById('metadataForm')?.addEventListener('submit', async e => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const res = await fetch('/metadata', { method: 'POST', body: formData });
  const data = await res.json();
  document.getElementById('metadataResult').textContent = JSON.stringify(data, null, 2);
});

// Set EEG Type
async function setEEGType() {
  const userId = document.getElementById('userIdType').value.trim();
  const type = document.getElementById('eegType').value;
  if (!userId || !type) return alert("Enter both User ID and EEG type.");

  const res = await fetch(`/eeg_types/${userId}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ eeg_type: type })
  });
  const data = await res.json();
  document.getElementById('typeResult').textContent = JSON.stringify(data, null, 2);
}

// Upload EEG Files
document.getElementById('eegForm')?.addEventListener('submit', async e => {
  e.preventDefault();
  const userId = document.getElementById('userIdEEG').value.trim();
  if (!userId) return alert("Enter User ID.");

  const formData = new FormData();
  const eegFiles = document.getElementById('eegFiles').files;
  for (let file of eegFiles) formData.append('eeg', file);

  const res = await fetch(`/eegs/${userId}`, { method: 'POST', body: formData });
  const data = await res.json();
  document.getElementById('eegResult').textContent = JSON.stringify(data, null, 2);
});
