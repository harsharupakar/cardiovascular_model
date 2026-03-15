document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();

    const form = document.getElementById('prediction-form');
    const isEverPregnantSelect = document.getElementById('is_ever_pregnant');
    const reproDetails = document.getElementById('pregnancy-complications');
    const chooseFileBtn = document.getElementById('choose-file-btn');
    const echoReportInput = document.getElementById('echo_report');
    const fileNameEl = document.getElementById('file-name');

    // Toggle reproductive complications based on pregnancy history
    isEverPregnantSelect.addEventListener('change', (e) => {
        if (e.target.value === "1") {
            reproDetails.classList.remove('hidden');
        } else {
            reproDetails.classList.add('hidden');
            // Reset values
            document.getElementById('gestational_diabetes').value = "0";
            document.getElementById('preeclampsia').value = "0";
            document.getElementById('preterm_birth').value = "0";
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitPrediction();
    });

    if (chooseFileBtn && echoReportInput && fileNameEl) {
        chooseFileBtn.addEventListener('click', () => echoReportInput.click());
        echoReportInput.addEventListener('change', () => {
            const file = echoReportInput.files && echoReportInput.files[0];
            const wrap = echoReportInput.closest('.file-upload-wrap');
            if (file) {
                fileNameEl.textContent = `Attached: ${file.name}`;
                if (wrap) wrap.classList.add('file-selected');
            } else {
                fileNameEl.textContent = 'No file chosen';
                if (wrap) wrap.classList.remove('file-selected');
            }
        });
    }
});

async function checkAPIHealth() {
    const statusEl = document.getElementById('api-status');
    try {
        const res = await fetch('http://localhost:8000/health');
        if (res.ok) {
            const data = await res.json();
            statusEl.textContent = `API Connected: v${data.model_version}`;
            statusEl.className = 'api-status healthy';
        } else {
            throw new Error('API Unhealthy');
        }
    } catch (e) {
        statusEl.textContent = 'Backend API is Offline';
        statusEl.className = 'api-status offline';
    }
}

async function submitPrediction() {
    const btn = document.getElementById('analyze-btn');
    const loader = document.getElementById('btn-loader');
    const btnText = btn.querySelector('span');

    // UI Loading State
    btnText.textContent = "Processing...";
    loader.classList.remove('hidden');
    btn.disabled = true;

    const numberOrDefault = (id, fallback) => {
        const raw = document.getElementById(id)?.value;
        const parsed = parseFloat(raw);
        return Number.isFinite(parsed) ? parsed : fallback;
    };

    const intOrDefault = (id, fallback) => {
        const raw = document.getElementById(id)?.value;
        const parsed = parseInt(raw, 10);
        return Number.isInteger(parsed) ? parsed : fallback;
    };

    // Build Payload (defaults ensure upload-only flow still works)
    const payload = {
        age: numberOrDefault('age', 28),
        BMI: numberOrDefault('BMI', 24),
        blood_pressure: numberOrDefault('blood_pressure', 120),
        glucose: numberOrDefault('glucose', 95),
        activity: numberOrDefault('activity', 3),
        cholesterol: numberOrDefault('cholesterol', 180),
        sleep_duration: numberOrDefault('sleep_duration', 7),
        alcohol: numberOrDefault('alcohol', 0),
        diet_pattern: intOrDefault('diet_pattern', 1),
        stress_level: intOrDefault('stress_level', 1),
        education: intOrDefault('education', 2),
        socioeconomic_status: intOrDefault('socioeconomic_status', 2),
        smoking: intOrDefault('smoking', 0),
        PCOS: intOrDefault('PCOS', 0),
        hypertension: intOrDefault('hypertension', 0),
        is_ever_pregnant: intOrDefault('is_ever_pregnant', 0),
        gestational_diabetes: intOrDefault('gestational_diabetes', 0),
        preeclampsia: intOrDefault('preeclampsia', 0),
        preterm_birth: intOrDefault('preterm_birth', 0)
    };

    const echoFile = document.getElementById('echo_report').files[0];
    if (!echoFile) {
        alert('Please upload an echocardiography PDF report.');
        btnText.textContent = "Analyze Patient Profile";
        loader.classList.add('hidden');
        btn.disabled = false;
        return;
    }

    const formData = new FormData();
    formData.append("patient_data", JSON.stringify(payload));
    formData.append("echo_report", echoFile);

    try {
        const response = await fetch('http://localhost:8000/predict_comprehensive', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);

    } catch (error) {
        alert("Prediction Failed. Ensure the backend API is running on localhost:8000.\n\nError: " + error.message);
    } finally {
        // Reset UI State
        btnText.textContent = "Analyze Patient Profile";
        loader.classList.add('hidden');
        btn.disabled = false;
    }
}

function renderResults(data) {
    document.getElementById('results-placeholder').classList.add('hidden');
    const content = document.getElementById('results-content');
    content.classList.remove('hidden');

    const riskEl = document.getElementById('res-risk-level');
    const probEl = document.getElementById('res-probability');
    const barEl = document.getElementById('res-probability-bar');
    const confEl = document.getElementById('res-confidence');
    const shapList = document.getElementById('res-shap-list');
    const fusionList = document.getElementById('res-fusion-breakdown');

    // 1. Text & Colors
    riskEl.textContent = `${data.risk_level} Risk`;
    riskEl.className = '';
    barEl.style.background = ''; // Clear inline styles to let css gradients take over

    if (data.risk_level === 'High') {
        riskEl.classList.add('risk-level-high');
        barEl.style.background = 'linear-gradient(90deg, #FF4757, #ff6b81)';
    } else if (data.risk_level === 'Moderate') {
        riskEl.classList.add('risk-level-moderate');
        barEl.style.background = 'linear-gradient(90deg, #FFA502, #eccc68)';
    } else {
        riskEl.classList.add('risk-level-low');
        barEl.style.background = 'linear-gradient(90deg, #2ED573, #7bed9f)';
    }

    // 2. Metrics
    const percent = Math.round(data.probability * 100);
    probEl.textContent = `${percent}%`;
    barEl.style.width = `${percent}%`;
    confEl.textContent = `CI \u00B1${data.confidence_interval}`;

    // 3. SHAP Factors map
    shapList.innerHTML = '';
    if (data.top_factors && data.top_factors.length > 0) {
        data.top_factors.forEach(factor => {
            const li = document.createElement('li');
            li.className = 'shap-item';
            
            const nameSpan = document.createElement('div');
            nameSpan.className = 'shap-name';
            nameSpan.textContent = formatFeatureName(factor.feature);

            const impactSpan = document.createElement('div');
            impactSpan.className = 'shap-impact ' + (factor.impact > 0 ? 'positive' : 'negative');
            impactSpan.textContent = formatImpact(factor.impact);

            li.appendChild(nameSpan);
            li.appendChild(impactSpan);
            shapList.appendChild(li);
        });
    } else {
        shapList.innerHTML = '<li>No SHAP data returned from model.</li>';
    }

    // 4. Echo Results
    const echoMetricsList = document.getElementById('res-echo-metrics');
    const echoRisksList = document.getElementById('res-echo-risks');
    
    echoMetricsList.innerHTML = '';
    const metricsForDisplay = data.echo_metrics_extracted_raw || data.echo_metrics;
    const defaultedFields = Array.isArray(data.echo_fields_defaulted) ? data.echo_fields_defaulted : [];
    if (metricsForDisplay) {
        for (const [key, value] of Object.entries(metricsForDisplay)) {
            const li = document.createElement('li');
            li.className = 'shap-item';
            const isDefaulted = defaultedFields.includes(key);
            const shown = (value === null || value === undefined) ? 'null' : value;
            const suffix = isDefaulted ? " <span style='opacity:0.7'>(missing in report)</span>" : '';
            li.innerHTML = `<div class='shap-name'>${formatFeatureName(key)}</div><div class='shap-impact'>${shown}${suffix}</div>`;
            echoMetricsList.appendChild(li);
        }
    }
    
    echoRisksList.innerHTML = '';
    if (data.echo_disease_risks) {
        for (const [key, value] of Object.entries(data.echo_disease_risks)) {
            const li = document.createElement('li');
            li.className = 'shap-item';
            const riskPct = Math.round(value * 100);
            const riskClass = riskPct > 50 ? 'negative' : 'positive'; 
            li.innerHTML = `<div class='shap-name'>${formatFeatureName(key)}</div><div class='shap-impact ${riskClass}'>${riskPct}% Risk</div>`;
            echoRisksList.appendChild(li);
        }
    }

    // 5. Fusion transparency
    if (fusionList) {
        fusionList.innerHTML = '';
        const rows = [
            { label: 'Lifestyle Model Risk', value: data.lifestyle_risk },
            { label: 'Echo Model Max Risk', value: data.echo_model_max_risk },
            { label: 'Echo Structural Risk', value: data.echo_structural_risk },
            { label: 'Final Combined Risk', value: data.probability }
        ];

        rows.forEach(row => {
            if (typeof row.value !== 'number' || Number.isNaN(row.value)) return;
            const pct = Math.round(row.value * 100);
            const li = document.createElement('li');
            li.className = 'shap-item';
            li.innerHTML = `<div class='shap-name'>${row.label}</div><div class='shap-impact ${pct >= 50 ? 'positive' : 'negative'}'>${pct}%</div>`;
            fusionList.appendChild(li);
        });

        if (data.shap_engine) {
            const li = document.createElement('li');
            li.className = 'shap-item';
            li.innerHTML = `<div class='shap-name'>SHAP Engine</div><div class='shap-impact'>${String(data.shap_engine)}</div>`;
            fusionList.appendChild(li);
        }

        if (data.echo_extraction_status) {
            const li = document.createElement('li');
            li.className = 'shap-item';
            li.innerHTML = `<div class='shap-name'>Echo Extraction Status</div><div class='shap-impact'>${String(data.echo_extraction_status)}</div>`;
            fusionList.appendChild(li);
        }

        if (Array.isArray(data.echo_fields_defaulted) && data.echo_fields_defaulted.length > 0) {
            const li = document.createElement('li');
            li.className = 'shap-item';
            li.innerHTML = `<div class='shap-name'>Defaulted Fields</div><div class='shap-impact negative'>${data.echo_fields_defaulted.join(', ')}</div>`;
            fusionList.appendChild(li);
        }

        if (data.echo_extraction_warning) {
            const li = document.createElement('li');
            li.className = 'shap-item';
            li.innerHTML = `<div class='shap-name'>Extraction Warning</div><div class='shap-impact negative'>${String(data.echo_extraction_warning)}</div>`;
            fusionList.appendChild(li);
        }

        if (!fusionList.children.length) {
            fusionList.innerHTML = '<li class="shap-item"><div class="shap-name">Fusion details unavailable</div><div class="shap-impact">-</div></li>';
        }
    }

    // Use disclaimer as recommendation
    if (data.disclaimer) {
        let recDiv = document.getElementById('res-recommendation');
        if (!recDiv) {
            recDiv = document.createElement('div');
            recDiv.id = 'res-recommendation';
            recDiv.style.marginTop = '15px';
            recDiv.style.padding = '10px';
            recDiv.style.background = 'rgba(255,165,2,0.1)';
            recDiv.style.border = '1px solid rgba(255,165,2,0.3)';
            recDiv.style.borderRadius = '5px';
            recDiv.style.color = '#eccc68';
            document.getElementById('results-content').appendChild(recDiv);
        }
        recDiv.textContent = 'A.I. Recommendation: ' + data.disclaimer;
    }
}

function formatFeatureName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatImpact(value) {
    const absVal = Math.abs(Number(value) || 0);
    if (absVal === 0) return '0';
    if (absVal < 0.001) return absVal.toExponential(2);
    if (absVal < 0.01) return absVal.toFixed(5);
    return absVal.toFixed(3);
}
