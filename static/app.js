// app.js — UI interactions and AJAX logic
$(function () {
  // state
  let state = {
    currentUserId: null,
    lastMetadata: null,
    lastUploadResults: null
  };

  // set year
  $('#year').text(new Date().getFullYear());

  // navigation
  window.showSection = function (id) {
    $('.panel').removeClass('active');
    $('#' + id).addClass('active');
    // update active nav link visual (optional)
  };

  // default show home
  showSection('home');

  // HEALTH CHECK
  $('#btn-health').click(function () {
    $('#health-box').addClass('hide').text('');
    $.get('/health')
      .done(function (res) {
        $('#health-box').removeClass('hide').html(
          `<div><strong>Server:</strong> <span class="status-ok">${escapeHtml(res.message || 'OK')}</span></div>
           <div class="panel-sub">Ready at <code>/</code> and endpoints operational.</div>`
        );
      })
      .fail(function (xhr) {
        $('#health-box').removeClass('hide').html(`<div class="status-bad">Server unreachable</div>`);
      });
  });

  // preview button — backend may not implement preview; provide guidance
  $('#btn-preview').click(function () {
    alert('This backend version does not implement a /preview route. Use metadata upload to inspect previews returned by server.');
  });

  // HARDWARE REGISTER
  $('#hardware-form').submit(function (e) {
    e.preventDefault();
    const device = $('#hw-device').val().trim();
    const type = $('#hw-type').val().trim();

    if (!device) {
      alert('Please enter a device name');
      return;
    }

    const payload = {
      hardware_id: device,
      type: type || 'unknown'
    };

    $.ajax({
      url: '/hardwares',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify(payload),
      success: function (res) {
        $('#hardware-result').html(`<div><strong>${escapeHtml(res.message || 'Registered')}</strong><div class="panel-sub">hardware_id: <code>${escapeHtml(res.hardware_id)}</code></div></div>`);
      },
      error: function (xhr) {
        $('#hardware-result').text('Hardware registration failed');
      }
    });
  });

  // METADATA UPLOAD
  $('#btn-load-sample').click(function () {
    // sample metadata JSON (single subject)
    const sample = [
      {"subject_id": 1, "Age": 57, "Gender": "M", "MMSE": 28},
      {"subject_id": 2, "Age": 63, "Gender": "F", "MMSE": 30}
    ];
    // create downloadable blob for user convenience (and set file input via DataTransfer)
    const blob = new Blob([JSON.stringify(sample, null, 2)], {type: 'application/json'});
    const file = new File([blob], 'sample_metadata.json', {type: 'application/json'});
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    $('#metadata-file')[0].files = dataTransfer.files;
    alert('Sample metadata loaded into file input. Click "Upload Metadata".');
  });

  $('#btn-upload-metadata').click(async function () {
    const fileInput = $('#metadata-file')[0];
    if (!fileInput.files || fileInput.files.length === 0) {
      alert('Choose a metadata JSON/CSV file first.');
      return;
    }
    const file = fileInput.files[0];

    const fd = new FormData();
    // backend expects form field named 'metadata'
    fd.append('metadata', file);

    // optionally send user_id to reuse
    if (state.currentUserId) fd.append('user_id', state.currentUserId);

    $('#metadata-result').text('Uploading metadata...');
    try {
      const res = await ajaxForm('/metadata', fd);
      if (res && res.user_id) {
        state.currentUserId = res.user_id;
        state.lastMetadata = res.metadata;
        $('#current-user').text(res.user_id);
      }
      $('#metadata-result').html(`<div><strong>${escapeHtml(res.message)}</strong>
        <div class="panel-sub">user_id: <code>${escapeHtml(res.user_id)}</code></div>
        <pre>${escapeHtml(JSON.stringify(res.metadata, null, 2))}</pre></div>`);
      // switch to declare type / upload panel for convenience
      showSection('upload');
    } catch (err) {
      $('#metadata-result').text(`Metadata upload failed: ${err.message || JSON.stringify(err)}`);
    }
  });

  // SET EEG TYPE
  $('#btn-set-type').click(function () {
    let userid = $('#type-userid').val().trim() || state.currentUserId;
    const type = $('#eeg-type-select').val();

    if (!userid) {
      alert('User ID required. Upload metadata first or paste user id.');
      return;
    }
    if (!type) {
      alert('Select an EEG type (edf/csv/set).');
      return;
    }

    $.ajax({
      url: '/eeg_types/' + encodeURIComponent(userid),
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ eeg_type: type }),
      success: function (res) {
        $('#metadata-result').html(`<div><strong>${escapeHtml(res.message)}</strong><pre>${escapeHtml(JSON.stringify(res.metadata, null, 2))}</pre></div>`);
        // update last metadata in state
        state.lastMetadata = res.metadata;
        $('#typeResult') && $('#typeResult').text(res.message);
        alert('EEG type set successfully.');
      },
      error: function (xhr) {
        let msg = 'Failed to set EEG type';
        if (xhr && xhr.responseJSON && xhr.responseJSON.error) msg = xhr.responseJSON.error;
        alert(msg);
      }
    });
  });

  $('#btn-clear-type').click(function () {
    $('#eeg-type-select').val('');
    $('#type-userid').val('');
  });

  // DROPZONE — file selection + drag & drop
  const $drop = $('#drop-zone');
  const $fileInput = $('#eeg-input');

  $drop.on('dragover', function (e) {
    e.preventDefault();
    $(this).addClass('dragover');
  });
  $drop.on('dragleave drop', function (e) {
    e.preventDefault();
    $(this).removeClass('dragover');
  });

  $drop.on('click', function () {
    $fileInput.trigger('click');
  });

  $fileInput.on('change', function () {
    const files = Array.from(this.files);
    if (files.length) {
      $drop.find('.dz-text').text(`${files.length} file(s) selected`);
    }
  });

  // UPLOAD EEG FILES
  $('#btn-upload-eeg').click(async function () {
    let userid = $('#upload-userid').val().trim() || state.currentUserId;
    if (!userid) {
      alert('User ID is required. Upload metadata first or paste user id here.');
      return;
    }
    const files = $('#eeg-input')[0].files;
    if (!files || files.length === 0) {
      alert('Select EEG files to upload (drag & drop or click the box).');
      return;
    }

    const fd = new FormData();
    // backend expects form field name 'eeg' (multiple)
    for (let i = 0; i < files.length; i++) {
      fd.append('eeg', files[i], files[i].name);
    }

    // show progress bar
    $('#upload-progress').removeClass('hide');
    setProgress(0);

    try {
      const res = await ajaxFormWithProgress(`/eegs/${encodeURIComponent(userid)}`, fd, function (pct) {
        setProgress(pct);
      });

      // success
      state.lastUploadResults = res.results || null;
      state.lastMetadata = res.metadata || null;
      state.currentUserId = res.user_id || state.currentUserId;

      $('#upload-result').html(`<div><strong>${escapeHtml(res.message)}</strong>
        <pre>${escapeHtml(JSON.stringify(res.results || res.metadata, null, 2))}</pre></div>`);
      $('#summary-user').text(state.currentUserId || '—');
      $('#current-user').text(state.currentUserId || '—');
      $('#metadata-result').html(`<div class="monospace"><pre>${escapeHtml(JSON.stringify(state.lastMetadata, null, 2))}</pre></div>`);

    } catch (err) {
      $('#upload-result').text('Upload failed: ' + (err.message || JSON.stringify(err)));
    } finally {
      setProgress(100);
      setTimeout(() => {
        $('#upload-progress').addClass('hide');
        setProgress(0);
      }, 700);
    }
  });

  // SUMMARY controls
  $('#btn-show-metadata').click(function () {
    if (!state.lastMetadata) {
      $('#summary-metadata').text('No metadata available. Upload metadata first.');
      return;
    }
    $('#summary-user').text(state.currentUserId || '—');
    $('#summary-metadata').html(`<pre>${escapeHtml(JSON.stringify(state.lastMetadata, null, 2))}</pre>`);
    $('#summary-results').html(state.lastUploadResults ? `<pre>${escapeHtml(JSON.stringify(state.lastUploadResults, null, 2))}</pre>` : 'No recent upload results');
    showSection('summary');
  });

  $('#btn-download-metadata').click(function () {
    if (!state.lastMetadata) { alert('No metadata to download.'); return; }
    const blob = new Blob([JSON.stringify(state.lastMetadata, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (state.currentUserId || 'metadata') + '_metadata.json';
    a.click();
    URL.revokeObjectURL(url);
  });

  // Helper: set progress
  function setProgress(pct) {
    $('.progress-bar').css('width', `${pct}%`);
  }

  // Helper: AJAX form helper returning promise (without progress)
  function ajaxForm(url, formData) {
    return new Promise(function (resolve, reject) {
      $.ajax({
        url: url,
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (res) { resolve(res); },
        error: function (xhr) {
          const err = xhr.responseJSON || xhr.responseText || xhr.statusText;
          reject(err);
        }
      });
    });
  }

  // Helper: AJAX with progress
  function ajaxFormWithProgress(url, formData, onProgress) {
    return new Promise(function (resolve, reject) {
      $.ajax({
        url: url,
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        xhr: function () {
          const myXhr = $.ajaxSettings.xhr();
          if (myXhr.upload && onProgress) {
            myXhr.upload.addEventListener('progress', function (e) {
              if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                onProgress(percent);
              }
            }, false);
          }
          return myXhr;
        },
        success: function (res) { resolve(res); },
        error: function (xhr) {
          const err = xhr.responseJSON || xhr.responseText || xhr.statusText;
          reject(err);
        }
      });
    });
  }

  // Escape HTML helper
  function escapeHtml(str) {
    if (typeof str !== 'string') str = JSON.stringify(str, null, 2);
    return str.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  }

});
