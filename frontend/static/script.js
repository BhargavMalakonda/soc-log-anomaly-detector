/**
 * AI Log Anomaly Detection Dashboard — Client Logic
 * Implements: Requirements 1.2, 1.8, 1.9, 3.2, 4.1, 4.2, 4.3, 4.6, 4.7, 4.8,
 *             5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4
 */

(function () {
  'use strict';

  // ─── Module-level state ────────────────────────────────────────────────────
  let currentFile = null;     // the File object from the input
  let resultRows = [];        // array of row objects from /detect
  let resultColumns = [];     // array of column name strings from /detect

  // ─── DOM references (resolved after DOMContentLoaded) ─────────────────────
  let spinner, errorBanner, errorMessage, errorClose;
  let previewSection, previewContainer;
  let controlsSection, detectBtn, filterToggle, downloadBtn;
  let contaminationSlider, contaminationInput;
  let summarySection, summaryTotal, summaryAnomalies, summaryHigh,
      summaryMedium, summaryLow, summaryVisible;
  let resultsSection, resultsThead, resultsTbody;
  let featureReportSection, featureReportTbody;
  let fileInput;

  // ─── Task 11.1 — Spinner & Error helpers ──────────────────────────────────

  /** Removes 'hidden' class from #spinner */
  function showSpinner() {
    spinner.classList.remove('hidden');
  }

  /** Adds 'hidden' class to #spinner */
  function hideSpinner() {
    spinner.classList.add('hidden');
  }

  /**
   * Sets #error-message text and removes 'hidden' from #error-banner.
   * @param {string} message
   */
  function showError(message) {
    errorMessage.textContent = message;
    errorBanner.classList.remove('hidden');
  }

  /** Adds 'hidden' to #error-banner */
  function hideError() {
    errorBanner.classList.add('hidden');
  }

  // ─── Task 11.2 — uploadFile ────────────────────────────────────────────────

  /**
   * Sends the file to POST /upload, renders the preview table on success.
   * @param {File} file
   */
  async function uploadFile(file) {
    currentFile = file;
    hideError();
    showSpinner();

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || 'Upload failed. Please try again.');
        return;
      }

      // Store preview data and render
      const columns = data.columns || [];
      const previewRows = data.preview_rows || [];

      renderTable(previewRows, columns, 'preview-tbody', 'preview-thead', false);

      previewSection.classList.remove('hidden');
      controlsSection.classList.remove('hidden');

    } catch (err) {
      showError('Network error: could not reach the server. Please check your connection.');
    } finally {
      hideSpinner();
    }
  }

  // ─── Task 11.3 — detectAnomalies ──────────────────────────────────────────

  /**
   * Re-sends the stored file to POST /detect, renders results on success.
   */
  async function detectAnomalies() {
    if (!currentFile) {
      showError('Please upload a file first.');
      return;
    }

    hideError();
    showSpinner();
    detectBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append('file', currentFile);
      formData.append('contamination', contaminationInput.value);

      const response = await fetch('/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || 'Detection failed. Please try again.');
        return;
      }

      resultRows = data.rows || [];
      resultColumns = data.columns || [];

      // Render results table
      renderTable(resultRows, resultColumns, 'results-tbody', 'results-thead', true);

      // Update summary counts
      const summary = data.summary || {};
      summaryTotal.textContent = summary.total ?? resultRows.length;
      summaryAnomalies.textContent = summary.anomalies ?? 0;
      summaryHigh.textContent = summary.high ?? 0;
      summaryMedium.textContent = summary.medium ?? 0;
      summaryLow.textContent = summary.low ?? 0;

      summarySection.classList.remove('hidden');
      resultsSection.classList.remove('hidden');
      downloadBtn.disabled = false;

      // Render the feature report collapsible
      if (data.feature_report && data.feature_report.length > 0) {
        renderFeatureReport(data.feature_report);
        featureReportSection.classList.remove('hidden');
      }

      // Apply current filter state
      applyFilter(filterToggle.checked);

    } catch (err) {
      showError('Network error: could not reach the server. Please check your connection.');
    } finally {
      hideSpinner();
      detectBtn.disabled = false;
    }
  }

  // ─── Task 11.4 — renderTable ──────────────────────────────────────────────

  /**
   * Builds thead and tbody DOM from rows and columns arrays.
   *
   * For the results table (isResults=true):
   *   - Applies .row-anomaly when anomaly_label === -1
   *   - Applies .row-normal  when anomaly_label === 1
   *   - Sets data-label attribute on each <tr>
   *   - Wraps severity cell in <span class="badge badge-{severity}">
   *
   * For the preview table (isResults=false): no row colouring.
   *
   * @param {Object[]} rows
   * @param {string[]} columns
   * @param {string} tbodyId
   * @param {string} theadId
   * @param {boolean} isResults
   */
  function renderTable(rows, columns, tbodyId, theadId, isResults) {
    const thead = document.getElementById(theadId);
    const tbody = document.getElementById(tbodyId);

    // If the preview table elements don't exist yet, create them inside the container
    let resolvedThead = thead;
    let resolvedTbody = tbody;

    if (!resolvedThead || !resolvedTbody) {
      // Build a fresh table for the preview container
      const table = document.createElement('table');
      table.className = 'preview-table';
      resolvedThead = document.createElement('thead');
      resolvedThead.id = theadId;
      resolvedTbody = document.createElement('tbody');
      resolvedTbody.id = tbodyId;
      table.appendChild(resolvedThead);
      table.appendChild(resolvedTbody);
      previewContainer.innerHTML = '';
      previewContainer.appendChild(table);
    }

    // Build thead
    resolvedThead.innerHTML = '';
    const headerRow = document.createElement('tr');
    columns.forEach(function (col) {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    });
    resolvedThead.appendChild(headerRow);

    // Build tbody
    resolvedTbody.innerHTML = '';
    rows.forEach(function (row) {
      const tr = document.createElement('tr');

      if (isResults) {
        const label = row.anomaly_label;
        if (label === -1) {
          tr.classList.add('row-anomaly');
        } else if (label === 1) {
          tr.classList.add('row-normal');
        }
        tr.dataset.label = String(label);
      }

      columns.forEach(function (col) {
        const td = document.createElement('td');
        const value = row[col];

        if (isResults && col === 'severity' && value != null) {
          const badge = document.createElement('span');
          badge.className = 'badge badge-' + String(value).toLowerCase();
          badge.textContent = value;
          td.appendChild(badge);
        } else {
          td.textContent = value != null ? value : '';
        }

        tr.appendChild(td);
      });

      resolvedTbody.appendChild(tr);
    });
  }

  // ─── Task 11.5 — renderFeatureReport ─────────────────────────────────────

  /**
   * Populates #feature-report-tbody with one row per FeatureReportEntry.
   * @param {Array<{original_column: string, action: string, produced_features: string[], drop_reason: string}>} report
   */
  function renderFeatureReport(report) {
    featureReportTbody.innerHTML = '';

    report.forEach(function (entry) {
      const tr = document.createElement('tr');

      // Original column
      const tdCol = document.createElement('td');
      tdCol.textContent = entry.original_column;
      tr.appendChild(tdCol);

      // Action badge
      const tdAction = document.createElement('td');
      const badge = document.createElement('span');
      badge.className = 'action-badge action-' + entry.action;
      badge.textContent = entry.action.replace(/_/g, ' ');
      tdAction.appendChild(badge);
      tr.appendChild(tdAction);

      // Produced features (pills)
      const tdFeatures = document.createElement('td');
      if (entry.produced_features && entry.produced_features.length > 0) {
        entry.produced_features.forEach(function (feat) {
          const pill = document.createElement('span');
          pill.className = 'feature-pill';
          pill.textContent = feat;
          tdFeatures.appendChild(pill);
        });
      } else {
        tdFeatures.textContent = '—';
      }
      tr.appendChild(tdFeatures);

      // Drop reason
      const tdReason = document.createElement('td');
      tdReason.textContent = entry.drop_reason || '—';
      tr.appendChild(tdReason);

      featureReportTbody.appendChild(tr);
    });
  }

  // ─── Task 11.6 — applyFilter ──────────────────────────────────────────────

  /**
   * Toggles row visibility in #results-tbody based on anomaly_label.
   * Updates #summary-visible with the count of visible rows.
   * @param {boolean} showAnomaliesOnly
   */
  function applyFilter(showAnomaliesOnly) {
    const rows = resultsTbody.querySelectorAll('tr');
    let visibleCount = 0;

    rows.forEach(function (tr) {
      if (showAnomaliesOnly && tr.dataset.label === '1') {
        tr.classList.add('hidden');
      } else {
        tr.classList.remove('hidden');
        visibleCount++;
      }
    });

    summaryVisible.textContent = visibleCount;
  }

  // ─── Task 11.6 — downloadCSV ──────────────────────────────────────────────

  /**
   * Collects visible rows from #results-tbody, serialises to CSV,
   * and triggers a browser download of anomaly_results.csv.
   */
  function downloadCSV() {
    // Collect column names from thead
    const thElements = resultsThead.querySelectorAll('th');
    const columns = Array.from(thElements).map(function (th) {
      return th.textContent;
    });

    // Collect visible rows
    const rows = resultsTbody.querySelectorAll('tr:not(.hidden)');

    /**
     * Escapes a single CSV cell value.
     * Wraps in quotes if the value contains commas, quotes, or newlines.
     * @param {string} value
     * @returns {string}
     */
    function escapeCell(value) {
      const str = value == null ? '' : String(value);
      if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
        return '"' + str.replace(/"/g, '""') + '"';
      }
      return str;
    }

    // Build CSV lines
    const lines = [];
    lines.push(columns.map(escapeCell).join(','));

    rows.forEach(function (tr) {
      const cells = tr.querySelectorAll('td');
      const rowValues = Array.from(cells).map(function (td) {
        // Use textContent (badge span text is included naturally)
        return escapeCell(td.textContent);
      });
      lines.push(rowValues.join(','));
    });

    const csvString = lines.join('\r\n');

    // Create Blob and trigger download
    const blob = new Blob([csvString], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'anomaly_results.csv';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  // ─── Task 11.7 — Wire up event listeners ──────────────────────────────────

  document.addEventListener('DOMContentLoaded', function () {
    // Resolve DOM references
    spinner = document.getElementById('spinner');
    errorBanner = document.getElementById('error-banner');
    errorMessage = document.getElementById('error-message');
    errorClose = document.getElementById('error-close');

    previewSection = document.getElementById('preview-section');
    previewContainer = document.getElementById('preview-container');

    controlsSection = document.getElementById('controls-section');
    detectBtn = document.getElementById('detect-btn');
    filterToggle = document.getElementById('filter-toggle');
    downloadBtn = document.getElementById('download-btn');
    contaminationSlider = document.getElementById('contamination-slider');
    contaminationInput = document.getElementById('contamination-input');

    // Keep slider and number input in sync
    contaminationSlider.addEventListener('input', function () {
      contaminationInput.value = this.value;
    });
    contaminationInput.addEventListener('input', function () {
      const v = parseFloat(this.value);
      if (!isNaN(v) && v >= 0.01 && v <= 0.30) {
        contaminationSlider.value = v;
      }
    });

    summarySection = document.getElementById('summary-section');
    summaryTotal = document.getElementById('summary-total');
    summaryAnomalies = document.getElementById('summary-anomalies');
    summaryHigh = document.getElementById('summary-high');
    summaryMedium = document.getElementById('summary-medium');
    summaryLow = document.getElementById('summary-low');
    summaryVisible = document.getElementById('summary-visible');

    resultsSection = document.getElementById('results-section');
    resultsThead = document.getElementById('results-thead');
    resultsTbody = document.getElementById('results-tbody');

    featureReportSection = document.getElementById('feature-report-section');
    featureReportTbody = document.getElementById('feature-report-tbody');

    fileInput = document.getElementById('file-input');

    // Upload button click → validate file → uploadFile
    document.getElementById('upload-btn').addEventListener('click', function () {
      const file = fileInput.files[0];
      if (!file) {
        showError('Please select a CSV file before uploading.');
        return;
      }
      uploadFile(file);
    });

    // Detect Anomalies button click
    detectBtn.addEventListener('click', function () {
      detectAnomalies();
    });

    // Show Anomalies Only toggle change
    filterToggle.addEventListener('change', function () {
      applyFilter(this.checked);
    });

    // Download Results button click
    downloadBtn.addEventListener('click', function () {
      downloadCSV();
    });

    // Error banner close button
    errorClose.addEventListener('click', function () {
      hideError();
    });

    // File input change → update label text to show selected filename
    fileInput.addEventListener('change', function () {
      const label = document.querySelector('label[for="file-input"]');
      if (label) {
        if (this.files && this.files[0]) {
          // Replace the text node (first child) with the filename
          const filename = this.files[0].name;
          // Find or create a text node before the input element
          let textNode = null;
          for (let i = 0; i < label.childNodes.length; i++) {
            if (label.childNodes[i].nodeType === Node.TEXT_NODE) {
              textNode = label.childNodes[i];
              break;
            }
          }
          if (textNode) {
            textNode.textContent = filename + '\n';
          } else {
            label.insertBefore(document.createTextNode(filename + '\n'), label.firstChild);
          }
        } else {
          // Reset label text
          let textNode = null;
          for (let i = 0; i < label.childNodes.length; i++) {
            if (label.childNodes[i].nodeType === Node.TEXT_NODE) {
              textNode = label.childNodes[i];
              break;
            }
          }
          if (textNode) {
            textNode.textContent = 'Choose CSV File\n';
          }
        }
      }
    });
  });

})();
