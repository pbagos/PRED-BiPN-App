

const API_BASE = '';

const ALPHA_METHODS = [
    'disparity filter',
    'doubly stochastic filter',
    'locally adaptive network sparsification filter',
    'marginal likelihood filter'
];

function CORRECTION_PARAM() {
    return { key: 'correction', label: 'Multiple-testing correction', type: 'select', value: 'fdr_bh',
             options: [['fdr_bh', 'Benjamini-Hochberg FDR'], ['bonferroni', 'Bonferroni'], ['none', 'None (raw p-value)']] };
}

function DIRECTED_PARAM() {
    return { key: 'directed', label: 'Network type', type: 'select', value: 'false',
             options: [['false', 'Undirected (A–B = B–A)'], ['true', 'Directed']] };
}

// shown for every method: the backend otherwise guesses from the first row
const COMMON_PARAMS = [
    { key: 'has_header', label: 'First row is a header', type: 'select', value: 'auto',
      options: [['auto', 'Auto-detect'], ['true', 'Yes'], ['false', 'No']] }
];

function paramSpecs(method) {
    return method ? [...(METHOD_PARAMS[method.toLowerCase()] || []), ...COMMON_PARAMS] : null;
}

const METHOD_PARAMS = {
    'noise corrected filter':   [DIRECTED_PARAM()],
    'doubly stochastic filter': [DIRECTED_PARAM()],
    'cosine similarity weighted projection':            [{ key: 'threshold', label: 'Similarity threshold (0–1)', value: 0.5, min: 0, max: 1, step: 0.01 }],
    'adamic adar similarity weighted projection':       [{ key: 'threshold', label: 'Similarity threshold', value: 0.5, min: 0, step: 0.01 }],
    'resource allocation similarity weighted projection': [{ key: 'threshold', label: 'Similarity threshold', value: 0.5, min: 0, step: 0.01 }],
    'jaccard similarity weighted projection':           [{ key: 'threshold', label: 'Jaccard threshold (0–1)', value: 0.0, min: 0, max: 1, step: 0.01 }],
    'kendall correlation weighted projection':          [CORRECTION_PARAM(), { key: 'alpha', label: 'Significance level α', value: 0.05, min: 0.0001, max: 1, step: 0.005 }],
    'spearman correlation weighted projection':         [CORRECTION_PARAM(), { key: 'alpha', label: 'Significance level α', value: 0.05, min: 0.0001, max: 1, step: 0.005 }],
    'pearson correlation weighted projection':          [CORRECTION_PARAM(), { key: 'alpha', label: 'Significance level α', value: 0.05, min: 0.0001, max: 1, step: 0.005 }],
    'odds ratio weighted projection': [
        CORRECTION_PARAM(),
        { key: 'alpha', label: 'Significance level α', value: 0.05, min: 0.0001, max: 1, step: 0.005 },
        { key: 'or_threshold', label: 'Odds ratio threshold', value: 1.0, min: 0, step: 0.1 }
    ],
    'high salience skeleton':  [{ key: 'threshold', label: 'Salience threshold (0–1)', value: 0.9, min: 0, max: 1, step: 0.01 }],
    'edge betweenness': [
        { key: 'factor', label: 'Keep edges with betweenness ≥ factor × mean', value: 1.0, min: 0, step: 0.1 },
        { key: 'weighted', label: 'Use edge weights (length = 1/weight)', type: 'select', value: 'true', options: [['true', 'Yes'], ['false', 'No']] },
        { key: 'sample_size', label: 'Sampled source nodes (0 = exact)', value: 0, min: 0, step: 1 }
    ],
    'hierarchical backbone': [
        { key: 'z_threshold',     label: 'Z-score threshold', value: 1.64, step: 0.01 },
        { key: 'alpha_percentile', label: 'Alpha percentile (1–100)', value: 95, min: 1, max: 100, step: 1 }
    ]
};

function renderDynamicParams(method) {
    const panel = document.getElementById('dynamicParamsPanel');
    if (!panel) return;
    const specs = paramSpecs(method);
    if (!specs) {
        panel.style.display = 'none';
        panel.innerHTML = '';
        return;
    }
    panel.innerHTML = specs.map(s => {
        const lbl = `<label class="param-label" for="dp_${s.key}">${s.label}:</label>`;
        if (s.type === 'select') {
            const opts = s.options.map(([v, t]) =>
                `<option value="${v}"${v === s.value ? ' selected' : ''}>${t}</option>`
            ).join('');
            return lbl + `<select id="dp_${s.key}" class="param-input">${opts}</select>`;
        }
        return lbl +
            `<input type="number" id="dp_${s.key}" class="param-input" value="${s.value}"` +
            `${s.min != null ? ` min="${s.min}"` : ''}${s.max != null ? ` max="${s.max}"` : ''} step="${s.step || 'any'}" />`;
    }).join('');
    panel.style.display = 'block';
}

function collectDynamicParams(method) {
    const specs = paramSpecs(method);
    const out = {};
    if (!specs) return out;
    for (const s of specs) {
        const el = document.getElementById('dp_' + s.key);
        if (!el) continue;
        if (s.type === 'select') {
            out[s.key] = el.value;
        } else if (el.value !== '' && !isNaN(parseFloat(el.value))) {
            out[s.key] = parseFloat(el.value);
        }
    }
    return out;
}

let latestTaskId = null;
let selectedMethod = null;
let selectedCategory = null;
let lastUploadedFormData = null;
let currentMethod = null;
let currentResultFile = null;

function showFieldError(inputId, message) {
    const input = document.getElementById(inputId);
    if (!input) return;

    input.classList.add('error');
    let existingMessage = input.nextElementSibling;
    if (existingMessage && existingMessage.classList.contains('error-message')) {
        existingMessage.remove();
    }

    const errorSpan = document.createElement('div');
    errorSpan.className = 'error-message';
    errorSpan.textContent = message;
    input.insertAdjacentElement('afterend', errorSpan);
}

function clearFieldErrors() {
    document.querySelectorAll('input').forEach(input => {
        input.classList.remove('error');
        let errorElem = input.nextElementSibling;
        if (errorElem && errorElem.classList.contains('error-message')) {
            errorElem.remove();
        }
    });
}

async function downloadFile(url, filename) {
    const response = await fetch(url, {
        method: 'GET',
        credentials: 'include'
    });

    if (!response.ok) {
        throw new Error('File download failed');
    }

    const blob = await response.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
}

function setupChooseMethod() {
    const projectionsBtn = document.getElementById('projectionsBtn');
    const backboneBtn = document.getElementById('backboneBtn');
    const projectionsOptions = document.getElementById('projectionsOptions');
    const backboneOptions = document.getElementById('backboneOptions');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const previewBox = document.getElementById('filePreviewBox');
    const continueBtn = document.getElementById('continueBtn');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const downloadBtn = document.getElementById('downloadBtn');
    const downloadMessage = document.getElementById('downloadMessage');

    continueBtn.style.display = 'none';

    function closeAllDropdowns() {
        if (projectionsOptions) projectionsOptions.style.display = 'none';
        if (backboneOptions) backboneOptions.style.display = 'none';
    }

    function onMethodSelected(item, sourceBtn) {
        selectedMethod = item.textContent.replace(/^\d+\.\s*/, '').trim();
        selectedCategory = (sourceBtn === backboneBtn) ? 'backbone' : 'projection';
        projectionsBtn.innerHTML = 'Bipartite Projections <span class="dropdown-arrow">▼</span>';
        backboneBtn.innerHTML = 'Backbone Extraction <span class="dropdown-arrow">▼</span>';
        sourceBtn.innerHTML = selectedMethod + ' <span class="dropdown-arrow">▼</span>';
        closeAllDropdowns();

        hideVisualization();
        hideAllScreens();

        previewBox.style.display = 'none';
        previewBox.textContent = '';
        fileInput.value = '';
        lastUploadedFormData = null;
        downloadSection.style.display = 'none';

        const polyaPanel = document.getElementById('polyaPanel');
        if (ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
            paramField.style.display = 'block';
        } else {
            paramField.style.display = 'none';
        }
        if (polyaPanel) {
            polyaPanel.style.display = selectedMethod.toLowerCase() === 'polya filter' ? 'block' : 'none';
        }
        const globalThresholdPanel = document.getElementById('globalThresholdPanel');
        if (globalThresholdPanel) {
            if (selectedMethod.toLowerCase() === 'global threshold filter') {
                globalThresholdPanel.style.display = 'block';
                const uploadedFile = sessionStorage.getItem('uploaded_file');
                if (uploadedFile) {
                    fetch(`${API_BASE}/get_file_columns?filename=${uploadedFile}`)
                        .then(r => r.json())
                        .then(data => {
                            if (data.success && typeof updateThresholdPanel === 'function') {
                                updateThresholdPanel(data.columns);
                            }
                        })
                        .catch(() => {});
                }
            } else {
                globalThresholdPanel.style.display = 'none';
            }
        }

        renderDynamicParams(selectedMethod);
    }

    projectionsBtn.addEventListener('click', e => {
        e.stopPropagation();
        const isOpen = projectionsOptions.style.display === 'block';
        closeAllDropdowns();
        if (!isOpen) projectionsOptions.style.display = 'block';
    });

    backboneBtn.addEventListener('click', e => {
        e.stopPropagation();
        const isOpen = backboneOptions.style.display === 'block';
        closeAllDropdowns();
        if (!isOpen) backboneOptions.style.display = 'block';
    });

    document.addEventListener('click', e => {
        if (!projectionsBtn.contains(e.target) && !projectionsOptions.contains(e.target) &&
            !backboneBtn.contains(e.target) && !backboneOptions.contains(e.target)) {
            closeAllDropdowns();
        }
    });

    document.querySelectorAll('#projectionsOptions li').forEach(item => {
        item.addEventListener('click', () => onMethodSelected(item, projectionsBtn));
    });

    document.querySelectorAll('#backboneOptions li').forEach(item => {
        item.addEventListener('click', () => onMethodSelected(item, backboneBtn));
    });

    uploadBtn.addEventListener('click', () => {
        if (!selectedMethod) {
            alert('Please select a method first.');
            return;
        }
        fileInput.click();
    });

    fileInput.addEventListener('change', async event => {
        const file = event.target.files[0];
        if (!file) {
            previewBox.style.display = 'none';
            continueBtn.style.display = 'none';
            lastUploadedFormData = null;
            return;
        }

        const allowedTypes = ['text/plain', 'text/csv'];
        const allowedExtensions = ['.txt', '.csv', '.tsv'];
        const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
            alert('Please select a .txt, .csv, or .tsv file');
            fileInput.value = '';
            previewBox.style.display = 'none';
            continueBtn.style.display = 'none';
            lastUploadedFormData = null;
            return;
        }

        previewBox.style.display = 'block';
        previewBox.textContent = `📄 ${file.name}\n\nLoading content...`;

        const reader = new FileReader();
        reader.onload = e => {
            const content = e.target.result;

            const lines = content.split('\n').slice(0, 10).join('\n');
            previewBox.textContent = `📄 ${file.name}\n\nFirst 10 lines:\n${lines}`;
        };
        reader.readAsText(file);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('method', selectedMethod);

        if (ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
            const alphaValue = document.getElementById('alphaInput')?.value || 0.05;
            formData.append('alpha', alphaValue);
        }

        lastUploadedFormData = formData;

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
                credentials: 'include'
            });

            const data = await response.json();
            if (data.success) {
                sessionStorage.setItem('uploaded_file', data.uploaded_file);
                alert('File uploaded successfully! You can now click "Run Method" to process it.');
                continueBtn.style.display = 'inline-block';
            } else {
                alert(data.message || 'File upload error. Please check your file format.');
                continueBtn.style.display = 'none';
                lastUploadedFormData = null;
            }
        } catch (err) {
            alert('Network error during upload. Please check your connection.');
            continueBtn.style.display = 'none';
            lastUploadedFormData = null;
            console.error(err);
        }
    });

continueBtn.addEventListener('click', async () => {
    const uploadedFile = sessionStorage.getItem('uploaded_file');
    if (!selectedMethod) {
        alert('Please select a method first.');
        return;
    }
    if (!uploadedFile) {
        alert('Please upload a file first.');
        return;
    }
    continueBtn.disabled = true;
    continueBtn.textContent = 'Processing...';
    downloadSection.style.display = 'none';

    const processingPanel = document.getElementById('processingPanel');
    const procMethodLabel = document.getElementById('procMethodLabel');
    const terminalOutput = document.getElementById('terminalOutput');
    if (processingPanel) {
        if (procMethodLabel) procMethodLabel.textContent = selectedMethod;
        if (terminalOutput) terminalOutput.innerHTML = '';
        processingPanel.style.display = 'block';
    }

    function appendLog(msg) {
        if (!terminalOutput) return;
        const line = document.createElement('div');
        const lc = msg.toLowerCase();
        if (lc.startsWith('[error]')) line.className = 'log-error';
        else if (lc.startsWith('[warning]') || lc.startsWith('[warn]')) line.className = 'log-warn';
        else if (lc.startsWith('[debug]')) line.className = 'log-debug';
        else line.className = 'log-info';
        line.textContent = msg;
        terminalOutput.appendChild(line);
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }

    try {
        let alphaValue = 0.05;
        if (ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
            const input = document.getElementById('alphaInput');
            if (input) alphaValue = parseFloat(input.value) || 0.05;
        }

        let requestData = {
            uploaded_file: uploadedFile,
            method: selectedMethod,
            alpha: alphaValue
        };

        if (selectedMethod.toLowerCase() === 'polya filter') {
            requestData.alpha = parseFloat(document.getElementById('polyaAlpha')?.value) || 0.05;
            const aVal = parseFloat(document.getElementById('polyaA')?.value);
            requestData.a_parameter = isNaN(aVal) ? -1 : aVal;
            requestData.correction_method = document.getElementById('polyaCorrection')?.value || 'fdr';
            requestData.directed = document.getElementById('polyaDirected')?.value || 'false';
        }

        if (selectedMethod.toLowerCase() === 'global threshold filter') {
            let params = null;
            const stored = sessionStorage.getItem('threshold_params');
            if (stored) {
                params = JSON.parse(stored);
            } else {

                const colSel = document.getElementById('thresholdColumn');
                const dirSel = document.getElementById('thresholdDirection');
                const valIn  = document.getElementById('thresholdValue');
                if (colSel && dirSel && valIn && valIn.value !== '') {
                    const dirRaw = dirSel.value;
                    params = {
                        column: colSel.value === 'auto' ? 'auto' : parseInt(colSel.value),
                        direction: dirRaw === 'greater' ? 'gte' : dirRaw === 'less' ? 'lte' : dirRaw,
                        value: parseFloat(valIn.value)
                    };
                }
            }
            if (params && !isNaN(params.value)) {
                const direction = params.direction === 'greater' ? 'gte' :
                                  params.direction === 'less' ? 'lte' :
                                  params.direction;
                requestData = {
                    ...requestData,
                    col_index: params.column === 'auto' ? 2 : params.column,
                    threshold: params.value,
                    direction: direction,
                    chunk_size: 50000
                };
            } else {
                alert('Please enter a threshold value.');
                continueBtn.disabled = false;
                continueBtn.textContent = 'Run Method';
                if (processingPanel) processingPanel.style.display = 'none';
                return;
            }
        }

        Object.assign(requestData, collectDynamicParams(selectedMethod));

        const bounded01 = ['cosine similarity weighted projection', 'jaccard similarity weighted projection'];
        if (bounded01.includes((selectedMethod || '').toLowerCase())) {
            const t = requestData.threshold;
            if (t == null || isNaN(t) || t < 0 || t > 1) {
                clearFieldErrors();
                showFieldError('dp_threshold', 'Threshold must be between 0 and 1.');
                continueBtn.disabled = false;
                continueBtn.textContent = 'Run Method';
                if (processingPanel) processingPanel.style.display = 'none';
                return;
            }
        }

        window.currentMethod = selectedMethod;

        const startResp = await fetch(`${API_BASE}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData),
            credentials: 'include'
        });
        const startData = await startResp.json();

        if (!startResp.ok) {
            throw new Error(startData.message || 'Failed to start processing');
        }

        const taskId = startData.task_id;
        appendLog(`[INFO] Task started: ${taskId}`);

        let seenLogs = 0;
        await new Promise((resolve, reject) => {
            const timer = setInterval(async () => {
                try {
                    const pollResp = await fetch(`${API_BASE}/task_status/${taskId}`, {
                        credentials: 'include'
                    });
                    const pollData = await pollResp.json();

                    const logs = pollData.logs || [];
                    for (let i = seenLogs; i < logs.length; i++) {
                        appendLog(logs[i]);
                    }
                    seenLogs = logs.length;

                    if (pollData.status === 'done') {
                        clearInterval(timer);
                        if (processingPanel) processingPanel.style.display = 'none';
                        resolve(pollData.result);
                    } else if (pollData.status === 'error') {
                        clearInterval(timer);
                        reject(new Error(pollData.error || 'Processing failed'));
                    }

                } catch (pollErr) {
                    clearInterval(timer);
                    reject(pollErr);
                }
            }, 800);
        }).then((result) => {
            const data = result;
            const methodName = window.currentMethod || selectedMethod;
            const files = data.files || {};
            let resultFiles = [];

            if (files.result_file) {
                resultFiles.push({name: files.result_file, desc: `${methodName} Result`});
                if (files.communities_file) {
                    resultFiles.push({name: files.communities_file, desc: `${methodName} Communities`});
                }
            } else if (files.result) {
                resultFiles.push({name: files.result, desc: `${methodName} Result`});
            } else if (files.node1_file && files.node2_file) {
                resultFiles.push({name: files.node1_file, desc: `${methodName} - Node1 Projection`});
                resultFiles.push({name: files.node2_file, desc: `${methodName} - Node2 Projection`});
            } else if (files.gene_file && files.disease_file) {
                resultFiles.push({name: files.gene_file, desc: `${methodName} - Gene Projection`});
                resultFiles.push({name: files.disease_file, desc: `${methodName} - Disease Projection`});
            } else {
                for (const key in files) {
                    if (typeof files[key] === 'string' && files[key].length > 0) {
                        resultFiles.push({name: files[key], desc: `${methodName} Result`});
                        break;
                    }
                }
            }

            if (resultFiles.length === 0) {
                throw new Error('No result files found in the response.');
            }

            currentResultFile = resultFiles;

            sessionStorage.setItem('result_file', resultFiles[0].name);
            if (resultFiles.length >= 2) {
                sessionStorage.setItem('result_file_2', resultFiles[1].name);
            }

            downloadMessage.textContent = resultFiles.length === 1
                ? `Processing completed. Result: ${resultFiles[0].name}`
                : `Processing completed. ${resultFiles.length} result files generated.`;

            downloadSection.style.display = 'block';
            continueBtn.style.display = 'none';

            if (window.updateComparisonButtons) {
                window.updateComparisonButtons();
            }
        });

    } catch (error) {
        console.error('Error during processing:', error);
        const processingPanel = document.getElementById('processingPanel');
        const terminalOutput = document.getElementById('terminalOutput');
        if (terminalOutput) {
            const errLine = document.createElement('div');
            errLine.className = 'log-error';
            errLine.textContent = `[ERROR] ${error.message}`;
            terminalOutput.appendChild(errLine);
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        } else {
            alert(`Error: ${error.message}`);
        }
    } finally {
        continueBtn.disabled = false;
        continueBtn.textContent = 'Run Method';
    }
});

    if (downloadBtn) {
        downloadBtn.addEventListener('click', async () => {
            if (!currentResultFile || currentResultFile.length === 0) {
                alert('No result file to download.');
                return;
            }

            downloadBtn.disabled = true;
            downloadBtn.textContent = 'Downloading...';

            try {

                for (const fileInfo of currentResultFile) {
                    await downloadFile(
                        `${API_BASE}/download?file=${encodeURIComponent(fileInfo.name)}`,
                        fileInfo.name
                    );
                }

                const completeMsg = document.createElement('div');
                completeMsg.style.cssText = `
                    margin-top: 10px;
                    padding: 10px;
                    background-color: rgba(76, 175, 80, 0.8);
                    color: white;
                    border-radius: 5px;
                    text-align: center;
                    font-family: monospace;
                `;
                completeMsg.textContent = 'Download completed successfully! You can now select a new method and upload another file.';
                methodPanel.appendChild(completeMsg);

                setTimeout(() => {
                    if (completeMsg.parentNode) {
                        completeMsg.parentNode.removeChild(completeMsg);
                    }
                }, 5000);

                sessionStorage.removeItem('uploaded_file');
                fileInput.value = '';
                previewBox.style.display = 'none';
                previewBox.textContent = '';
                downloadSection.style.display = 'none';
                continueBtn.style.display = 'none';
                currentResultFile = null;

                const pBtn2 = document.getElementById('projectionsBtn');
                const bBtn2 = document.getElementById('backboneBtn');
                if (pBtn2) pBtn2.innerHTML = 'Bipartite Projections <span class="dropdown-arrow">▼</span>';
                if (bBtn2) bBtn2.innerHTML = 'Backbone Extraction <span class="dropdown-arrow">▼</span>';
                selectedMethod = null;
                selectedCategory = null;

            } catch (error) {
                alert(`Download error: ${error.message}`);
            } finally {
                downloadBtn.disabled = false;
                downloadBtn.textContent = 'Download Result';
            }
        });
    }
}

let isGuideVisible = false;

function showGuideContent() {
    const mainPanel = document.getElementById('mainPanel');
    const methodPanel = document.getElementById('methodPanel');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const globalThresholdPanel = document.getElementById('globalThresholdPanel');
    const visualizationSection = document.getElementById('visualizationSection');
    const continueBtn = document.getElementById('continueBtn');
    const uploadBtn = document.getElementById('uploadBtn');

    if (methodPanel) methodPanel.style.display = 'none';
    if (paramField) paramField.style.display = 'none';
    if (downloadSection) downloadSection.style.display = 'none';
    if (globalThresholdPanel) globalThresholdPanel.style.display = 'none';
    const polyaPanelGuide = document.getElementById('polyaPanel');
    if (polyaPanelGuide) polyaPanelGuide.style.display = 'none';
    const dpGuide = document.getElementById('dynamicParamsPanel');
    if (dpGuide) dpGuide.style.display = 'none';
    if (visualizationSection) visualizationSection.style.display = 'none';

    if (continueBtn) continueBtn.style.display = 'none';
    if (uploadBtn) uploadBtn.style.display = 'none';
    const methodDropdownGroup = document.getElementById('methodDropdownGroup');
    if (methodDropdownGroup) methodDropdownGroup.style.display = 'none';

    const previewBox = document.getElementById('filePreviewBox');
    if (previewBox) previewBox.style.display = 'none';

    const projOpts = document.getElementById('projectionsOptions');
    if (projOpts) projOpts.style.display = 'none';
    const bkbOpts = document.getElementById('backboneOptions');
    if (bkbOpts) bkbOpts.style.display = 'none';

    const existingGuide = document.getElementById('guideContent');
    if (existingGuide) existingGuide.remove();

    const helpContent = document.createElement('div');
    helpContent.id = 'guideContent';
    helpContent.className = 'help-main-content';
    helpContent.innerHTML = `
        <div class="help-main-container">
            <div class="how-to-use-header" style="text-align:center; margin-bottom:2rem;">
                <h1>Guide</h1>
                <p>How to use PRED-BiPN &amp; Scientific Background</p>
            </div>
            <div class="how-to-use-steps">
                <div class="how-to-use-step"><div class="step-number">1</div><div class="step-content"><h3>Choose a method</h3><p>Click <strong>Bipartite Projections</strong> or <strong>Backbone Extraction</strong> to expand the method list and select one of the 23 available algorithms.</p><div class="step-tip"><strong>Tip:</strong> Projection methods turn a bipartite network into a monopartite one; backbone methods prune a weighted network. The relevant parameter fields appear automatically once a method is selected.</div></div></div>
                <div class="how-to-use-step"><div class="step-number">2</div><div class="step-content"><h3>Upload your network file</h3><p>Click <strong>Upload file</strong> and select your .csv, .tsv, or .txt file. A preview of the first 10 rows appears automatically.</p><div class="step-tip"><strong>Supported:</strong> 2-column bipartite (node type A → node type B, e.g. gene → disease, author → paper, user → product) or 3-column weighted (Source → Target → Weight).</div></div></div>
                <div class="how-to-use-step"><div class="step-number">3</div><div class="step-content"><h3>Configure parameters and execute</h3><p>Set the parameters shown for your method, then click <strong>Run Method</strong>.</p><div class="step-tip"><strong>Correlation & odds-ratio methods:</strong> choose a significance level α (default 0.05) and a multiple-testing correction &mdash; Benjamini-Hochberg FDR (default), Bonferroni, or None. <strong>Similarity & salience methods</strong> use a threshold between 0 and 1. Stricter settings keep fewer edges.</div></div></div>
                <div class="how-to-use-step"><div class="step-number">4</div><div class="step-content"><h3>Compare and download results</h3><p>Once done, <strong>Download Result</strong> (TSV) or <strong>Export as PNG</strong>. For backbone-extraction methods you can also <strong>Compare Networks</strong> to view the original vs filtered network side by side.</p></div></div>
                <div class="how-to-use-step"><div class="step-number">5</div><div class="step-content"><h3>Reset or start new analysis</h3><p>Use the <strong>Tool</strong> button in the top bar to reset and start fresh.</p></div></div>
            </div>
            <hr style="margin: 2rem 0; border-color: rgba(255,255,255,0.2);">

            <!-- Input format -->
            <div class="help-main-section">
                <h3>⨀ Input Format</h3>
                <p>
                    The header row is detected automatically; set <strong>First row is a header</strong> to Yes or No
                    if your first row is misread. Node names are trimmed of surrounding spaces, and names that differ only
                    by a trailing underscore (e.g. <code>X_OF_BREAST</code> and <code>X_OF_BREAST_</code>) are reported in
                    the log, since they are counted as different nodes.
                </p>
                <p>
                    Projections are treated as undirected (A&ndash;B is the same edge as B&ndash;A). Noise Corrected,
                    Doubly Stochastic and Pólya filters offer a <strong>Directed</strong> option for directed networks.
                    Hierarchical Backbone needs the original two-column bipartite file, not a projection.
                    The Marginal Likelihood and Pólya filters model integer counts; non-integer weights are
                    truncated or rounded, and a warning appears in the log.
                </p>
            </div>

            <!-- Edge Betweenness -->
            <div class="help-main-section">
                <h3>⨀ Edge Betweenness</h3>
                <p>
                    The edge betweenness of an edge is the number of shortest paths between node pairs that pass
                    through it (Girvan &amp; Newman, 2002), computed once for all edges with Brandes' algorithm
                    (Brandes, 2001). Weights are similarities, so each weight <em>w</em> becomes a length 1/<em>w</em>.
                    An edge is kept if its betweenness &ge; <em>factor</em> &times; mean betweenness.
                </p>
                <p>
                    <strong>factor</strong> (default 1): larger values give sparser backbones.
                    <strong>Use edge weights</strong>: No sets every length to 1.
                    <strong>Sampled source nodes</strong>: 0 computes exact betweenness; a positive number estimates it
                    from that many random source nodes, for very large networks.
                    Output: <code>source  target  weight  edge_betweenness</code>, sorted by betweenness.
                </p>
                <p class="citation">
                    Brandes, U. (2001). A faster algorithm for betweenness centrality. <em>Journal of Mathematical
                    Sociology</em>, 25(2), 163&ndash;177.
                    <a href="https://doi.org/10.1080/0022250X.2001.9990249" target="_blank" rel="noopener">https://doi.org/10.1080/0022250X.2001.9990249</a><br>
                    Girvan, M., &amp; Newman, M. E. J. (2002). Community structure in social and biological networks.
                    <em>PNAS</em>, 99(12), 7821&ndash;7826.
                    <a href="https://doi.org/10.1073/pnas.122653799" target="_blank" rel="noopener">https://doi.org/10.1073/pnas.122653799</a>
                </p>
            </div>

            <!-- Data Sources -->
            <div class="help-main-section">
                <h3>⨀ Data Sources</h3>
                <p>
                    The gene-disease association data used to validate the methods
                    comes from:
                </p>
                <p class="citation">
                    Kontou, P. I., Pavlopoulou, A., Dimou, N. L., Pavlopoulos, G. A., &amp; Bagos, P. G.
                    (2016). <em>Network analysis of genes and their association with diseases</em>.
                    Gene, 590(1), 68&ndash;78.
                    <a href="https://doi.org/10.1016/j.gene.2016.05.044" target="_blank" rel="noopener">https://doi.org/10.1016/j.gene.2016.05.044</a>
                </p>
                <p>
                    The tool is not limited to this dataset &mdash; any bipartite or weighted edge list works.
                </p>
            </div>

            <!-- Technical Specifications -->
            <div class="help-main-section">
                <h3>⨯ Technical Specifications</h3>
                <div class="tech-specs">
                    <div class="spec-item">
                        <strong>Languages:</strong> Python, JavaScript, HTML/CSS
                    </div>
                    <div class="spec-item">
                        <strong>Libraries:</strong> NetworkX, igraph, Pandas, SciPy, NumPy, Scikit-learn
                    </div>
                    <div class="spec-item">
                        <strong>Parallel Processing:</strong> Multiprocessing support for large-scale networks
                    </div>
                    <div class="spec-item">
                        <strong>Export Formats:</strong> TSV
                    </div>
                </div>
            </div>

            <button id="backToToolFromGuideBtn" class="back-to-tool-btn">← Back to Tool</button>
        </div>
    `;

    mainPanel.insertBefore(helpContent, methodPanel);

    isGuideVisible = true;

    const backBtn = document.getElementById('backToToolFromGuideBtn');
    if (backBtn) {
        backBtn.addEventListener('click', hideGuideContent);
    }
}

function hideGuideContent() {
    const mainPanel = document.getElementById('mainPanel');
    const methodPanel = document.getElementById('methodPanel');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const helpContent = document.getElementById('guideContent');
    const continueBtn = document.getElementById('continueBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const previewBox = document.getElementById('filePreviewBox');

    if (helpContent) {
        helpContent.remove();
    }

    if (methodPanel) methodPanel.style.display = 'flex';
    if (continueBtn) continueBtn.style.display = 'inline-block';
    if (uploadBtn) uploadBtn.style.display = 'inline-block';
    const methodDropdownGroup = document.getElementById('methodDropdownGroup');
    if (methodDropdownGroup) methodDropdownGroup.style.display = 'flex';

    if (selectedMethod && ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
        if (paramField) paramField.style.display = 'block';
    } else {
        if (paramField) paramField.style.display = 'none';
    }
    const polyaPanelHide = document.getElementById('polyaPanel');
    if (polyaPanelHide) {
        polyaPanelHide.style.display = selectedMethod && selectedMethod.toLowerCase() === 'polya filter' ? 'block' : 'none';
    }

    const dpHide = document.getElementById('dynamicParamsPanel');
    if (dpHide) dpHide.style.display = selectedMethod ? 'block' : 'none';

    if (downloadSection) downloadSection.style.display = 'none';
    if (previewBox) previewBox.style.display = 'none';

    isGuideVisible = false;
}

function setupGuideButton() {
    const helpBtn = document.getElementById('helpBtn');

    if (helpBtn) {
        helpBtn.addEventListener('click', function() {
            if (isGuideVisible) {
                hideGuideContent();
            } else {
                showGuideContent();
            }
        });
    }

    const infoBtn = document.getElementById('infoBtn');
    if (infoBtn) {
        infoBtn.addEventListener('click', function() {
            if (isInfoVisible) {
                hideInfoContent();
            } else {
                if (isGuideVisible) hideGuideContent();
                showInfoContent();
            }
        });
    }
}

let isInfoVisible = false;

function showInfoContent() {
    const mainPanel = document.getElementById('mainPanel');
    const methodPanel = document.getElementById('methodPanel');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const globalThresholdPanel = document.getElementById('globalThresholdPanel');
    const visualizationSection = document.getElementById('visualizationSection');
    const continueBtn = document.getElementById('continueBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const methodDropdownGroup = document.getElementById('methodDropdownGroup');
    const previewBox = document.getElementById('filePreviewBox');

    if (methodPanel) methodPanel.style.display = 'none';
    if (paramField) paramField.style.display = 'none';
    if (downloadSection) downloadSection.style.display = 'none';
    if (globalThresholdPanel) globalThresholdPanel.style.display = 'none';
    const polyaPanelInfo = document.getElementById('polyaPanel');
    if (polyaPanelInfo) polyaPanelInfo.style.display = 'none';
    const dpInfo = document.getElementById('dynamicParamsPanel');
    if (dpInfo) dpInfo.style.display = 'none';
    if (visualizationSection) visualizationSection.style.display = 'none';
    if (continueBtn) continueBtn.style.display = 'none';
    if (uploadBtn) uploadBtn.style.display = 'none';
    if (methodDropdownGroup) methodDropdownGroup.style.display = 'none';
    if (previewBox) previewBox.style.display = 'none';

    const existing = document.getElementById('infoContent');
    if (existing) existing.remove();

    const infoDiv = document.createElement('div');
    infoDiv.id = 'infoContent';
    infoDiv.className = 'help-main-content';
    infoDiv.innerHTML = `
        <div class="help-main-container">
            <div class="how-to-use-header" style="text-align:center; margin-bottom:2rem;">
                <h1>Credits</h1>
                <p>The people behind PRED-BiPN</p>
            </div>

            <div class="help-main-section">
                <h3>Development</h3>
                <ul class="bibliography-list">
                    <li><strong>Efstathios Voulgaris</strong> &mdash; Department of Computer Science and Biomedical Informatics, University of Thessaly, Lamia, 35131, Greece</li>
                    <li><strong>Ioannis Charatsiaris</strong> &mdash; Department of Computer Science and Biomedical Informatics, University of Thessaly, Lamia, 35131, Greece</li>
                </ul>
            </div>

            <div class="help-main-section">
                <h3>Academic Supervision</h3>
                <ul class="bibliography-list">
                    <li><strong>Pantelis Bagos, Professor</strong> &mdash; Department of Computer Science and Biomedical Informatics, University of Thessaly, Lamia, 35131, Greece &mdash; <a href="mailto:pbagos@compgen.org">pbagos@compgen.org</a></li>
                    <li><strong>Panagiota I. Kontou, Assistant Professor</strong> &mdash; Department of Mathematics, University of Thessaly, 35131 Lamia, Greece &mdash; <a href="mailto:pkontou@uth.gr">pkontou@uth.gr</a></li>
                </ul>
            </div>

            <button id="backToToolFromInfoBtn" class="back-to-tool-btn">← Back to Tool</button>
        </div>
    `;

    mainPanel.insertBefore(infoDiv, methodPanel);
    isInfoVisible = true;

    const backBtn = document.getElementById('backToToolFromInfoBtn');
    if (backBtn) {
        backBtn.addEventListener('click', hideInfoContent);
    }
}

function hideInfoContent() {
    const mainPanel = document.getElementById('mainPanel');
    const methodPanel = document.getElementById('methodPanel');
    const paramField = document.getElementById('paramField');
    const downloadSection = document.getElementById('downloadSection');
    const continueBtn = document.getElementById('continueBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const methodDropdownGroup = document.getElementById('methodDropdownGroup');
    const infoDiv = document.getElementById('infoContent');

    if (infoDiv) infoDiv.remove();

    if (methodPanel) methodPanel.style.display = 'flex';
    if (continueBtn) continueBtn.style.display = 'inline-block';
    if (uploadBtn) uploadBtn.style.display = 'inline-block';
    if (methodDropdownGroup) methodDropdownGroup.style.display = 'flex';

    if (selectedMethod && ALPHA_METHODS.includes(selectedMethod.toLowerCase())) {
        if (paramField) paramField.style.display = 'block';
    } else {
        if (paramField) paramField.style.display = 'none';
    }
    const polyaPanelInfoHide = document.getElementById('polyaPanel');
    if (polyaPanelInfoHide) {
        polyaPanelInfoHide.style.display = selectedMethod && selectedMethod.toLowerCase() === 'polya filter' ? 'block' : 'none';
    }

    const dpInfoHide = document.getElementById('dynamicParamsPanel');
    if (dpInfoHide) dpInfoHide.style.display = selectedMethod ? 'block' : 'none';

    if (downloadSection) downloadSection.style.display = 'none';

    isInfoVisible = false;
}

function createVisualizationSection() {
    const mainPanel = document.getElementById('mainPanel');
    if (!mainPanel) return;

    if (document.getElementById('visualizationSection')) return;

    const visSection = document.createElement('div');
    visSection.id = 'visualizationSection';
    visSection.className = 'visualization-section';
    visSection.style.display = 'none';
    visSection.innerHTML = `
        <div class="vis-header">
            <div class="vis-title">
                <span class="vis-icon">⌗</span>
                <h3>Network Topology Comparison</h3>
            </div>
            <button id="closeVisSection" class="vis-close">×</button>
        </div>
        <div class="vis-body">
            <div id="visLoading" class="vis-loading">
                <div class="vis-spinner"></div>
                <p>Generating network visualization...</p>
            </div>
            <div id="visContent" style="display: none;">
                <div class="comparison-wrapper">
                    <div class="network-panel">
                        <div class="panel-header before">
                            <div class="panel-title">
                                <span class="panel-icon">⊡</span>
                                <span>Original Network</span>
                            </div>
                            <div class="panel-stats" id="statsBefore">-</div>
                        </div>
                        <div id="networkBefore" class="network-display"></div>
                        <div class="panel-metrics" id="metricsBefore">
                            <div class="metric-item">
                                <span class="metric-symbol">ω</span>
                                <span class="metric-value">-</span>
                            </div>
                        </div>
                    </div>
                    <div class="vs-divider">
                        <span>↯</span>
                    </div>
                    <div class="network-panel">
                        <div class="panel-header after">
                            <div class="panel-title">
                                <span class="panel-icon">⌗</span>
                                <span>Filtered Network</span>
                            </div>
                            <div class="panel-stats" id="statsAfter">-</div>
                        </div>
                        <div id="networkAfter" class="network-display"></div>
                        <div class="panel-metrics" id="metricsAfter">
                            <div class="metric-item">
                                <span class="metric-symbol">ω</span>
                                <span class="metric-value">-</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="comparison-footer">
                    <div id="reductionInfo" class="reduction-badge"></div>
                    <button id="exportVisBtn" class="vis-export-btn">
                        <span>⤥</span> Export
                    </button>
                </div>
            </div>
        </div>
    `;

    mainPanel.appendChild(visSection);

    document.getElementById('closeVisSection')?.addEventListener('click', () => {
        visSection.style.display = 'none';
        const hideBtn = document.getElementById('hideComparisonBtn');
        if (hideBtn) hideBtn.style.display = 'none';
    });

    document.getElementById('exportVisBtn')?.addEventListener('click', () => {
        const img = document.querySelector('#networkBefore img');
        if (img && img.src) {
            const link = document.createElement('a');
            link.download = `network_comparison_${new Date().toISOString().slice(0,19)}.png`;
            link.href = img.src;
            link.click();
        }
    });
}

async function showInlineComparison(beforeFile, afterFile, method) {
    console.log('showInlineComparison called with:', { beforeFile, afterFile, method });

    if (!beforeFile || !afterFile) {
        console.error('Missing files:', { beforeFile, afterFile });
        alert('Missing files for comparison. Please ensure both original and result files are available.');
        return;
    }

    if (!document.getElementById('visualizationSection')) {
        createVisualizationSection();
    }

    const visSection = document.getElementById('visualizationSection');
    const visLoading = document.getElementById('visLoading');
    const visContent = document.getElementById('visContent');
    const statsBefore = document.getElementById('statsBefore');
    const statsAfter = document.getElementById('statsAfter');
    const reductionInfo = document.getElementById('reductionInfo');
    const networkBefore = document.getElementById('networkBefore');
    const networkAfter = document.getElementById('networkAfter');
    const metricsBefore = document.getElementById('metricsBefore');
    const metricsAfter = document.getElementById('metricsAfter');
    const hideBtn = document.getElementById('hideComparisonBtn');

    visSection.style.display = 'block';
    visLoading.style.display = 'flex';
    visContent.style.display = 'none';

    if (hideBtn) hideBtn.style.display = 'inline-block';

    visSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    try {
        const response = await fetch(`${API_BASE}/visualize/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                before_file: beforeFile,
                after_file: afterFile,
                method: method
            }),
            credentials: 'include'
        });

        const data = await response.json();
        console.log('Comparison response:', data);

        if (data.success) {

            statsBefore.innerHTML = `${data.stats_before.nodes} nodes | ${data.stats_before.edges} edges`;
            statsAfter.innerHTML = `${data.stats_after.nodes} nodes | ${data.stats_after.edges} edges`;

            if (metricsBefore) {
                metricsBefore.innerHTML = `
                    <div class="metric-item">
                        <span class="metric-symbol">ω</span>
                        <span class="metric-label">Density</span>
                        <span class="metric-value">${data.stats_before.density.toFixed(4)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-symbol">δ</span>
                        <span class="metric-label">Edges / node</span>
                        <span class="metric-value">${(data.stats_before.edges / data.stats_before.nodes).toFixed(2)}</span>
                    </div>
                `;
            }

            if (metricsAfter) {
                metricsAfter.innerHTML = `
                    <div class="metric-item">
                        <span class="metric-symbol">ω</span>
                        <span class="metric-label">Density</span>
                        <span class="metric-value">${data.stats_after.density.toFixed(4)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-symbol">δ</span>
                        <span class="metric-label">Edges / node</span>
                        <span class="metric-value">${(data.stats_after.edges / data.stats_after.nodes).toFixed(2)}</span>
                    </div>
                `;
            }

            const reduction = data.reduction.toFixed(1);
            const edgesRemoved = data.stats_before.edges - data.stats_after.edges;
            reductionInfo.innerHTML = `⤥ ${reduction}% · ${edgesRemoved} edges removed`;

            networkBefore.innerHTML = '';
            networkAfter.innerHTML = '';

            const imgBefore = document.createElement('img');
            imgBefore.src = `data:image/png;base64,${data.image_before}`;
            imgBefore.alt = 'Original Network';
            imgBefore.style.maxWidth = '100%';
            imgBefore.style.height = 'auto';
            imgBefore.style.borderRadius = '8px';
            imgBefore.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
            networkBefore.appendChild(imgBefore);

            const imgAfter = document.createElement('img');
            imgAfter.src = `data:image/png;base64,${data.image_after}`;
            imgAfter.alt = 'Filtered Network';
            imgAfter.style.maxWidth = '100%';
            imgAfter.style.height = 'auto';
            imgAfter.style.borderRadius = '8px';
            imgAfter.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
            networkAfter.appendChild(imgAfter);

            visLoading.style.display = 'none';
            visContent.style.display = 'block';
        } else {
            throw new Error(data.message || 'Visualization failed');
        }
    } catch (error) {
        console.error('Comparison error:', error);
        visLoading.innerHTML = `
            <div style="color: #c0392b; text-align: center; padding: 2rem;">
                <strong>Error:</strong> ${error.message}<br><br>
                Make sure the backend server is running on ${API_BASE}
            </div>
        `;
    }
}

function setupComparisonButtons() {
    const compareBtn = document.getElementById('compareNetworksInlineBtn');
    const hideBtn = document.getElementById('hideComparisonBtn');

    if (!compareBtn) return { updateButtonsState: () => {} };

    function updateButtonsState() {
        const uploadedFile = sessionStorage.getItem('uploaded_file');
        let resultFile = null;

        if (currentResultFile && currentResultFile.length > 0) {
            resultFile = currentResultFile[0].name;
        } else if (sessionStorage.getItem('result_file')) {
            resultFile = sessionStorage.getItem('result_file');
        }

        console.log('Update buttons - Uploaded:', uploadedFile);
        console.log('Update buttons - Result:', resultFile);

        if (selectedCategory !== 'backbone') {
            compareBtn.style.display = 'none';
            if (hideBtn) hideBtn.style.display = 'none';
            const vis = document.getElementById('visualizationSection');
            if (vis) vis.style.display = 'none';
            return;
        }
        compareBtn.style.display = 'inline-block';

        if (uploadedFile && resultFile) {
            compareBtn.disabled = false;
            compareBtn.style.opacity = '1';
            compareBtn.style.cursor = 'pointer';
            compareBtn.style.pointerEvents = 'auto';
        } else {
            compareBtn.disabled = true;
            compareBtn.style.opacity = '0.5';
            compareBtn.style.cursor = 'not-allowed';
            compareBtn.style.pointerEvents = 'none';
        }
    }

    compareBtn.addEventListener('click', () => {
        if (compareBtn.disabled) {
            alert('Please wait for processing to complete.');
            return;
        }

        const currentUploaded = sessionStorage.getItem('uploaded_file');
        let currentResult = null;

        if (currentResultFile && currentResultFile.length > 0) {
            currentResult = currentResultFile[0].name;
        } else if (sessionStorage.getItem('result_file')) {
            currentResult = sessionStorage.getItem('result_file');
        }

        if (currentUploaded && currentResult) {
            showInlineComparison(currentUploaded, currentResult, selectedMethod);
            if (hideBtn) hideBtn.style.display = 'inline-block';
        } else {
            alert('Both original and result files are required for comparison.');
        }
    });

    if (hideBtn) {
        hideBtn.addEventListener('click', () => {
            const visSection = document.getElementById('visualizationSection');
            if (visSection) {
                visSection.style.display = 'none';
                hideBtn.style.display = 'none';
            }
        });

        hideBtn.style.display = 'none';
    }

    updateButtonsState();

    return { updateButtonsState };
}

let globalThresholdParams = {
    column: null,
    direction: 'greater',
    value: 0.5,
    columnNames: []
};

async function detectColumnsForThreshold(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/detect_columns`, {
            method: 'POST',
            body: formData,
            credentials: 'include'
        });

        const data = await response.json();
        if (data.success) {
            return data.columns;
        }
        return [];
    } catch (error) {
        console.error('Column detection error:', error);
        return [];
    }
}

function updateThresholdPanel(columns) {
    const columnSelect = document.getElementById('thresholdColumn');
    const columnPreview = document.getElementById('columnPreview');
    const detectedColumnsSpan = document.getElementById('detectedColumns');

    if (!columnSelect) return;

    columnSelect.innerHTML = '<option value="auto">Auto-detect (first numeric column)</option>';

    columns.forEach((col, idx) => {
        const option = document.createElement('option');
        option.value = idx;
        option.textContent = `${col.name} (${col.type})`;
        columnSelect.appendChild(option);
    });

    if (columns.length > 0) {
        detectedColumnsSpan.textContent = columns.map(c => `${c.name} (${c.type})`).join(', ');
        columnPreview.style.display = 'block';
    } else {
        detectedColumnsSpan.textContent = 'No numeric columns detected';
        columnPreview.style.display = 'block';
    }
}

function setupGlobalThreshold() {
    const methodOptions = document.querySelectorAll('#projectionsOptions li, #backboneOptions li');
    const globalThresholdPanel = document.getElementById('globalThresholdPanel');
    const applyBtn = document.getElementById('applyThresholdBtn');
    const columnSelect = document.getElementById('thresholdColumn');
    const directionSelect = document.getElementById('thresholdDirection');
    const valueInput = document.getElementById('thresholdValue');

    methodOptions.forEach(option => {
        option.addEventListener('click', () => {
            const methodText = option.textContent.toLowerCase();
            if (methodText.includes('global threshold')) {
                globalThresholdPanel.style.display = 'block';

                const uploadedFile = sessionStorage.getItem('uploaded_file');
                if (uploadedFile) {
                    fetch(`${API_BASE}/get_file_columns?filename=${uploadedFile}`)
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                updateThresholdPanel(data.columns);
                            }
                        });
                }
            } else {
                globalThresholdPanel.style.display = 'none';
            }
        });
    });

    if (applyBtn) {
        applyBtn.addEventListener('click', () => {
            const selectedColumn = columnSelect.value;
            const direction = directionSelect.value;
            const value = parseFloat(valueInput.value);

            const backendDirection = direction === 'greater' ? 'gte' : 'lte';

            globalThresholdParams = {
                column: selectedColumn === 'auto' ? 'auto' : parseInt(selectedColumn),
                direction: backendDirection,
                value: value
            };

            sessionStorage.setItem('threshold_params', JSON.stringify(globalThresholdParams));

            console.log('Threshold parameters saved:', globalThresholdParams);
            alert(`Threshold filter configured!\nColumn: ${selectedColumn === 'auto' ? 'Auto-detect' : columnSelect.options[columnSelect.selectedIndex].text}\nDirection: ${direction === 'greater' ? '≥' : '≤'} ${value}`);
        });
    }
}

function resetToMainPage() {
    console.log('Resetting to main page...');

    selectedMethod = null;
    selectedCategory = null;
    currentResultFile = null;
    currentMethod = null;
    lastUploadedFormData = null;

    sessionStorage.removeItem('uploaded_file');
    sessionStorage.removeItem('result_file');
    sessionStorage.removeItem('result_file_2');
    sessionStorage.removeItem('threshold_params');
    sessionStorage.removeItem('selected_method');

    const pBtn = document.getElementById('projectionsBtn');
    const bBtn = document.getElementById('backboneBtn');
    if (pBtn) pBtn.innerHTML = 'Bipartite Projections <span class="dropdown-arrow">▼</span>';
    if (bBtn) bBtn.innerHTML = 'Backbone Extraction <span class="dropdown-arrow">▼</span>';

    const projOpts = document.getElementById('projectionsOptions');
    const bkbOpts = document.getElementById('backboneOptions');
    if (projOpts) projOpts.style.display = 'none';
    if (bkbOpts) bkbOpts.style.display = 'none';

    const paramField = document.getElementById('paramField');
    if (paramField) {
        paramField.style.display = 'none';
    }

    const globalThresholdPanel = document.getElementById('globalThresholdPanel');
    if (globalThresholdPanel) {
        globalThresholdPanel.style.display = 'none';
    }
    const polyaPanelReset = document.getElementById('polyaPanel');
    if (polyaPanelReset) {
        polyaPanelReset.style.display = 'none';
    }
    const dpReset = document.getElementById('dynamicParamsPanel');
    if (dpReset) {
        dpReset.style.display = 'none';
    }

    const downloadSection = document.getElementById('downloadSection');
    if (downloadSection) {
        downloadSection.style.display = 'none';
    }

    const visualizationSection = document.getElementById('visualizationSection');
    if (visualizationSection) {
        visualizationSection.style.display = 'none';
    }

    const methodPanel = document.getElementById('methodPanel');
    if (methodPanel) {
        methodPanel.style.display = 'flex';
    }

    const continueBtn = document.getElementById('continueBtn');
    if (continueBtn) {
        continueBtn.style.display = 'none';
        continueBtn.disabled = false;
        continueBtn.textContent = 'Run Method';
    }

    const previewBox = document.getElementById('filePreviewBox');
    if (previewBox) {
        previewBox.style.display = 'none';
        previewBox.textContent = '';
    }

    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.value = '';
    }

    if (isGuideVisible) {
        hideGuideContent();
    }
    if (isInfoVisible) {
        hideInfoContent();
    }

    if (window.updateComparisonButtons) {
        window.updateComparisonButtons();
    }

    const downloadMessage = document.getElementById('downloadMessage');
    if (downloadMessage) {
        downloadMessage.textContent = '';
    }

    const alphaInput = document.getElementById('alphaInput');
    if (alphaInput) {
        alphaInput.value = '0.05';
    }

    console.log('Reset completed. Application is ready for new analysis.');

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function hideAllScreens() {
    if (isGuideVisible) {
        hideGuideContent();
    }
    if (isInfoVisible) {
        hideInfoContent();
    }
}

function hideVisualization() {
    const visSection = document.getElementById('visualizationSection');
    const hideBtn = document.getElementById('hideComparisonBtn');

    if (visSection) {
        visSection.style.display = 'none';
    }
    if (hideBtn) {
        hideBtn.style.display = 'none';
    }
}

document.addEventListener('DOMContentLoaded', () => {

    setupChooseMethod();

    setupGuideButton();

    const { updateButtonsState } = setupComparisonButtons();
    window.updateComparisonButtons = updateButtonsState;

    setupGlobalThreshold();

    const mainPageBtn = document.getElementById('mainPageBtn');
    if (mainPageBtn) {
        mainPageBtn.addEventListener('click', resetToMainPage);
    }

    console.log('PRED-BiPN application initialized');

    setTimeout(() => {
        const dynamicText = document.getElementById('dynamic-text');
        if (dynamicText) {
            dynamicText.style.opacity = '0.3';
            dynamicText.style.transition = 'opacity 1s ease';
        }
    }, 5000);
});

