/**
 * static/js/dashboard.js
 * Coordinates UI state, dataset loading, model training, and visualization updates.
 * Automatically uses the Python REST backend (`app.py`) when available, and seamlessly
 * falls back to the separated local training module (`model_trainer.js`) when opened
 * directly as an HTML file or via VS Code Live Server.
 */
(function () {
  const state = {
    dataset: null,
    rawCSVText: null,
    activePredictor: null,
    axisX: 0,
    axisY: 1,
    backendAvailable: false
  };

  function setStatus(text, isBusy = false) {
    const dot = document.getElementById("statusDot");
    const lbl = document.getElementById("statusText");
    if (lbl) lbl.textContent = `Status: ${text}`;
    if (dot) {
      dot.className = isBusy
        ? "w-2 h-2 rounded-full bg-amber-500 animate-pulse"
        : "w-2 h-2 rounded-full bg-emerald-500";
    }
  }

  function getConfigFromUI() {
    return {
      modelType: document.getElementById("selectModelType").value,
      lr: parseFloat(document.getElementById("inputLR").value),
      epochs: parseInt(document.getElementById("inputEpochs").value, 10),
      knnK: parseInt(document.getElementById("inputKNN").value, 10),
      treeDepth: parseInt(document.getElementById("inputDepth").value, 10),
      standardize: document.getElementById("chkStandardize").checked,
      axisX: state.axisX,
      axisY: state.axisY
    };
  }

  async function runTraining() {
    if (!state.dataset) return;
    setStatus("Training...", true);

    const cfg = getConfigFromUI();

    // Try Python backend first if running over http/https
    if (state.backendAvailable) {
      try {
        const res = await fetch("/api/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_type: cfg.modelType,
            hyperparams: {
              learning_rate: cfg.lr,
              epochs: cfg.epochs,
              n_neighbors: cfg.knnK,
              max_depth: cfg.treeDepth,
              threshold: 0.5
            },
            standardize: cfg.standardize,
            axis_x: cfg.axisX,
            axis_y: cfg.axisY
          })
        });
        if (res.ok) {
          const json = await res.json();
          if (json.status === "ok") {
            state.backendAvailable = true;
            updateDashboardUI(json.data);
            setStatus(`Trained (Test Acc: ${(json.data.metrics.test_accuracy * 100).toFixed(1)}%)`);
            return;
          }
        }
      } catch (_) {
        state.backendAvailable = false;
      }
    }

    // Fallback to separated local JS ModelTrainer module
    const result = window.ModelTrainer.trainPipeline(state.dataset, cfg);
    state.activePredictor = result.predictSingle;
    updateDashboardUI(result.responsePayload);
    setStatus(`Trained (Test Acc: ${(result.responsePayload.metrics.test_accuracy * 100).toFixed(1)}%)`);
  }

  async function generateNewSyntheticDataset() {
    const preset = document.getElementById("selectPreset").value;
    const samples = parseInt(document.getElementById("inputSamples").value, 10);
    const noise = parseFloat(document.getElementById("inputNoise").value);

    state.dataset = window.DataGenerator.generateSynthetic(preset, samples, noise);
    state.axisX = 0;
    state.axisY = Math.min(1, state.dataset.featureNames.length - 1);

    if (state.backendAvailable) {
      try {
        const cfg = getConfigFromUI();
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            preset,
            n_samples: samples,
            noise,
            model_type: cfg.modelType,
            hyperparams: {
              learning_rate: cfg.lr,
              epochs: cfg.epochs,
              n_neighbors: cfg.knnK,
              max_depth: cfg.treeDepth
            },
            standardize: cfg.standardize
          })
        });
        if (res.ok) {
          const json = await res.json();
          if (json.status === "ok") {
            state.backendAvailable = true;
            updateDashboardUI(json.data);
            setStatus(`Trained (Test Acc: ${(json.data.metrics.test_accuracy * 100).toFixed(1)}%)`);
            return;
          }
        }
      } catch (_) {
        state.backendAvailable = false;
      }
    }

    await runTraining();
  }

  function updateDashboardUI(data) {
    const ds = data.dataset_summary;
    const m = data.metrics;

    document.getElementById("summaryRows").textContent = ds.total_rows;
    document.getElementById("summaryFeatures").textContent = ds.feature_names.length;
    document.getElementById("summaryClasses").textContent = `0: ${ds.class_counts.class_0} | 1: ${ds.class_counts.class_1}`;

    document.getElementById("metricTestAcc").textContent = `${(m.test_accuracy * 100).toFixed(1)}%`;
    document.getElementById("metricTrainAcc").textContent = `Train: ${(m.train_accuracy * 100).toFixed(1)}%`;
    document.getElementById("metricPrecision").textContent = m.precision.toFixed(3);
    document.getElementById("metricRecall").textContent = m.recall.toFixed(3);
    document.getElementById("metricF1").textContent = m.f1_score.toFixed(3);
    document.getElementById("metricAUC").textContent = `ROC AUC: ${m.roc_auc.toFixed(3)}`;

    document.getElementById("cmTN").textContent = m.confusion_matrix.tn;
    document.getElementById("cmFP").textContent = m.confusion_matrix.fp;
    document.getElementById("cmFN").textContent = m.confusion_matrix.fn;
    document.getElementById("cmTP").textContent = m.confusion_matrix.tp;

    populateAxes(ds.feature_names, data.boundary_grid.axis_x, data.boundary_grid.axis_y);
    window.Visualizer.renderBoundaryCanvas(data.boundary_grid, ds.class_names);
    window.Visualizer.renderLossChart(data.loss_history);
    window.Visualizer.renderROCChart(data.roc_curve, m.roc_auc);
    window.Visualizer.renderWeightsChart(ds.feature_names, data.weights);

    renderPreviewTable(ds.feature_names, ds.target_name, ds.preview_rows);
    renderInferenceSliders(data.feature_ranges);
  }

  function populateAxes(featureNames, axX, axY) {
    const selX = document.getElementById("selectAxisX");
    const selY = document.getElementById("selectAxisY");
    selX.innerHTML = "";
    selY.innerHTML = "";

    featureNames.forEach((name, idx) => {
      const optX = document.createElement("option");
      optX.value = idx;
      optX.textContent = name;
      if (idx === axX) optX.selected = true;
      selX.appendChild(optX);

      const optY = document.createElement("option");
      optY.value = idx;
      optY.textContent = name;
      if (idx === axY) optY.selected = true;
      selY.appendChild(optY);
    });
  }

  function renderPreviewTable(featureNames, targetName, rows) {
    const headerRow = document.getElementById("previewHeaderRow");
    const bodyRows = document.getElementById("previewBodyRows");
    headerRow.innerHTML = "";
    bodyRows.innerHTML = "";

    const cols = [...featureNames, targetName];
    cols.forEach((c) => {
      const th = document.createElement("th");
      th.className = "px-3 py-2 font-semibold";
      th.textContent = c;
      headerRow.appendChild(th);
    });

    rows.forEach((r) => {
      const tr = document.createElement("tr");
      cols.forEach((c) => {
        const td = document.createElement("td");
        td.className = "px-3 py-1.5 text-slate-700";
        td.textContent = r[c];
        tr.appendChild(td);
      });
      bodyRows.appendChild(tr);
    });
  }

  function renderInferenceSliders(featureRanges) {
    const container = document.getElementById("inferenceSliders");
    container.innerHTML = "";

    featureRanges.forEach((fr) => {
      const step = ((fr.max - fr.min) / 100) || 0.05;
      const div = document.createElement("div");
      div.className = "bg-slate-50 p-2.5 rounded-md border border-slate-200";
      div.innerHTML = `
        <div class="flex justify-between text-xs mb-1">
          <span class="font-medium text-slate-700">${fr.name}</span>
          <span id="lblInf_${fr.index}" class="font-mono font-semibold text-slate-900">${fr.mean.toFixed(2)}</span>
        </div>
        <input type="range" data-idx="${fr.index}" class="infSlider w-full accent-blue-600 cursor-pointer" min="${fr.min}" max="${fr.max}" step="${step.toFixed(4)}" value="${fr.mean}">
      `;
      container.appendChild(div);
    });

    document.querySelectorAll(".infSlider").forEach((sl) => {
      sl.addEventListener("input", (e) => {
        const idx = e.target.getAttribute("data-idx");
        document.getElementById(`lblInf_${idx}`).textContent = parseFloat(e.target.value).toFixed(2);
        updateLivePrediction();
      });
    });

    updateLivePrediction();
  }

  async function updateLivePrediction() {
    const sliders = Array.from(document.querySelectorAll(".infSlider"));
    if (!sliders.length) return;
    const features = sliders.map((s) => parseFloat(s.value));

    if (state.backendAvailable) {
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ features })
        });
        if (res.ok) {
          const json = await res.json();
          if (json.status === "ok") {
            applyPredictionToUI(json.prediction);
            return;
          }
        }
      } catch (_) {}
    }

    if (state.activePredictor) {
      const pred = state.activePredictor(features);
      applyPredictionToUI(pred);
    }
  }

  function applyPredictionToUI(p) {
    const badge = document.getElementById("predBadge");
    badge.textContent = p.predicted_label;
    badge.className = p.predicted_class === 1
      ? "px-2 py-0.5 rounded text-xs font-semibold bg-rose-500 text-white"
      : "px-2 py-0.5 rounded text-xs font-semibold bg-sky-500 text-white";

    const p0 = (p.prob_class_0 * 100).toFixed(1);
    const p1 = (p.prob_class_1 * 100).toFixed(1);
    document.getElementById("predProb0").textContent = `Class 0: ${p0}%`;
    document.getElementById("predProb1").textContent = `Class 1: ${p1}%`;
    document.getElementById("barProb0").style.width = `${p0}%`;
    document.getElementById("barProb1").style.width = `${p1}%`;
  }

  function exportActiveCSV() {
    if (!state.dataset) return;
    const { rows, featureNames, targetName } = state.dataset;
    const header = [...featureNames, targetName].join(",");
    const lines = rows.map((r) => [...r.X.map((v) => Number(v.toFixed(4))), r.y].join(","));
    const csv = [header, ...lines].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "dataset.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  window.addEventListener("DOMContentLoaded", () => {
    // Mode tabs
    const tabSyn = document.getElementById("tabBtnSynthetic");
    const tabCSV = document.getElementById("tabBtnCSV");
    const panelSyn = document.getElementById("panelSynthetic");
    const panelCSV = document.getElementById("panelCSV");

    tabSyn.addEventListener("click", () => {
      tabSyn.className = "px-2.5 py-1 rounded bg-white text-slate-900 font-medium shadow-sm";
      tabCSV.className = "px-2.5 py-1 rounded text-slate-600 hover:text-slate-900";
      panelSyn.classList.remove("hidden");
      panelCSV.classList.add("hidden");
    });

    tabCSV.addEventListener("click", () => {
      tabCSV.className = "px-2.5 py-1 rounded bg-white text-slate-900 font-medium shadow-sm";
      tabSyn.className = "px-2.5 py-1 rounded text-slate-600 hover:text-slate-900";
      panelCSV.classList.remove("hidden");
      panelSyn.classList.add("hidden");
    });

    // Sliders
    document.getElementById("inputSamples").addEventListener("input", (e) => {
      document.getElementById("valSamples").textContent = e.target.value;
    });
    document.getElementById("inputNoise").addEventListener("input", (e) => {
      document.getElementById("valNoise").textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById("inputLR").addEventListener("input", (e) => {
      document.getElementById("valLR").textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById("inputEpochs").addEventListener("input", (e) => {
      document.getElementById("valEpochs").textContent = e.target.value;
    });
    document.getElementById("inputKNN").addEventListener("input", (e) => {
      document.getElementById("valKNN").textContent = e.target.value;
    });
    document.getElementById("inputDepth").addEventListener("input", (e) => {
      document.getElementById("valDepth").textContent = e.target.value;
    });

    document.getElementById("selectPreset").addEventListener("change", generateNewSyntheticDataset);
    document.getElementById("btnGenerateData").addEventListener("click", generateNewSyntheticDataset);
    document.getElementById("btnExportCSV").addEventListener("click", exportActiveCSV);
    document.getElementById("btnTrainModel").addEventListener("click", runTraining);

    document.getElementById("selectModelType").addEventListener("change", (e) => {
      const isLog = e.target.value === "logistic" || e.target.value === "poly_logistic";
      document.getElementById("boxParamsLogistic").classList.toggle("hidden", !isLog);
      document.getElementById("boxParamsOther").classList.toggle("hidden", isLog);
      runTraining();
    });

    // CSV File Input
    document.getElementById("fileInputCSV").addEventListener("change", async (e) => {
      if (!e.target.files || !e.target.files[0]) return;
      const text = await e.target.files[0].text();
      try {
        const { headers } = window.DataGenerator.parseRawCSV(text);
        state.rawCSVText = text;
        const selTarget = document.getElementById("selectTargetCol");
        selTarget.innerHTML = "";
        headers.forEach((h, i) => {
          const opt = document.createElement("option");
          opt.value = h;
          opt.textContent = h;
          if (i === headers.length - 1) opt.selected = true;
          selTarget.appendChild(opt);
        });
        document.getElementById("csvColumnSelector").classList.remove("hidden");
        state.dataset = window.DataGenerator.parseCSVToDataset(text, headers[headers.length - 1]);
        state.axisX = 0;
        state.axisY = Math.min(1, state.dataset.featureNames.length - 1);
        await runTraining();
      } catch (err) {
        setStatus(`CSV Error: ${err.message}`, true);
      }
    });

    document.getElementById("btnApplyUploadedCSV").addEventListener("click", async () => {
      if (!state.rawCSVText) return;
      const targetCol = document.getElementById("selectTargetCol").value;
      state.dataset = window.DataGenerator.parseCSVToDataset(state.rawCSVText, targetCol);
      state.axisX = 0;
      state.axisY = Math.min(1, state.dataset.featureNames.length - 1);
      await runTraining();
    });

    // Sample CSV buttons
    document.querySelectorAll(".btnSampleCSV").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const key = btn.getAttribute("data-sample");
        state.dataset = window.DataGenerator.getSampleDataset(key);
        state.axisX = 0;
        state.axisY = Math.min(1, state.dataset.featureNames.length - 1);
        await runTraining();
      });
    });

    // Axis selectors
    document.getElementById("selectAxisX").addEventListener("change", (e) => {
      state.axisX = parseInt(e.target.value, 10);
      runTraining();
    });
    document.getElementById("selectAxisY").addEventListener("change", (e) => {
      state.axisY = parseInt(e.target.value, 10);
      runTraining();
    });

    // Evaluation Chart Tabs
    const views = {
      loss: document.getElementById("viewChartLoss"),
      cm: document.getElementById("viewChartCM"),
      roc: document.getElementById("viewChartROC"),
      weights: document.getElementById("viewChartWeights")
    };

    document.querySelectorAll(".tabChartBtn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.getAttribute("data-chart");
        document.querySelectorAll(".tabChartBtn").forEach((b) => {
          b.className = "tabChartBtn px-2.5 py-1 rounded text-slate-600 hover:bg-slate-100";
        });
        btn.className = "tabChartBtn px-2.5 py-1 rounded bg-slate-900 text-white font-medium";
        Object.keys(views).forEach((k) => views[k].classList.toggle("hidden", k !== key));
      });
    });

    // Detect if served by the Python backend (app.py on port 5000) and run initial training
    if (window.location.protocol.startsWith("http") && window.location.port === "5000") {
      state.backendAvailable = true;
    }
    generateNewSyntheticDataset();
  });
})();
