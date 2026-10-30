/**
 * static/js/visualizer.js
 * Dedicated visualization module for rendering the 2D Decision Boundary Canvas
 * and Chart.js evaluation plots (Loss Curve, ROC Curve, Feature Weights).
 */
window.Visualizer = (function () {
  const charts = {
    loss: null,
    roc: null,
    weights: null
  };

  function renderBoundaryCanvas(grid, classNames) {
    const canvas = document.getElementById("boundaryCanvas");
    if (!canvas || !grid) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const res = grid.resolution;
    const cellW = W / res;
    const cellH = H / res;

    for (let r = 0; r < res; r++) {
      const rowIdx = res - 1 - r;
      for (let c = 0; c < res; c++) {
        const prob = grid.probs[rowIdx][c];
        if (prob < 0.5) {
          const intensity = (0.5 - prob) / 0.5;
          ctx.fillStyle = `rgba(14, 165, 233, ${0.22 + intensity * 0.52})`;
        } else {
          const intensity = (prob - 0.5) / 0.5;
          ctx.fillStyle = `rgba(244, 63, 94, ${0.22 + intensity * 0.52})`;
        }
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);

        if (Math.abs(prob - 0.5) < 0.025) {
          ctx.fillStyle = "rgba(251, 191, 36, 0.85)";
          ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
        }
      }
    }

    // Subtle grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo((W / 4) * i, 0);
      ctx.lineTo((W / 4) * i, H);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, (H / 4) * i);
      ctx.lineTo(W, (H / 4) * i);
      ctx.stroke();
    }

    // Scatter points
    const spanX = (grid.x_max - grid.x_min) || 1;
    const spanY = (grid.y_max - grid.y_min) || 1;

    grid.points.forEach((pt) => {
      const cx = ((pt.x - grid.x_min) / spanX) * W;
      const cy = H - ((pt.y - grid.y_min) / spanY) * H;
      ctx.beginPath();
      ctx.arc(cx, cy, 4.2, 0, 2 * Math.PI);
      ctx.fillStyle = pt.label === 1 ? "#f43f5e" : "#0ea5e9";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.3;
      ctx.fill();
      ctx.stroke();
    });

    if (classNames && classNames.length >= 2) {
      document.getElementById("legendClass0").textContent = classNames[0];
      document.getElementById("legendClass1").textContent = classNames[1];
    }
  }

  function renderLossChart(lossHistory) {
    const ctx = document.getElementById("canvasLossChart");
    if (!ctx || typeof Chart === "undefined") return;
    if (charts.loss) charts.loss.destroy();

    charts.loss = new Chart(ctx, {
      type: "line",
      data: {
        labels: lossHistory.map((item) => `Ep ${item.epoch}`),
        datasets: [{
          label: "Cross-Entropy Loss",
          data: lossHistory.map((item) => item.loss),
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.12)",
          fill: true,
          tension: 0.25,
          pointRadius: lossHistory.length > 50 ? 0 : 2,
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { maxTicksLimit: 8 } }
        }
      }
    });
  }

  function renderROCChart(rocCurve, aucVal) {
    const ctx = document.getElementById("canvasROCChart");
    if (!ctx || typeof Chart === "undefined") return;
    if (charts.roc) charts.roc.destroy();

    charts.roc = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: `ROC Curve (AUC = ${aucVal.toFixed(3)})`,
            data: rocCurve,
            borderColor: "#2563eb",
            backgroundColor: "rgba(37, 99, 235, 0.12)",
            fill: true,
            pointRadius: 1.5,
            borderWidth: 2
          },
          {
            label: "Random Guess (0.500)",
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: "#94a3b8",
            borderDash: [5, 5],
            pointRadius: 0,
            borderWidth: 1.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate" } },
          y: { type: "linear", min: 0, max: 1, title: { display: true, text: "True Positive Rate" } }
        }
      }
    });
  }

  function renderWeightsChart(featureNames, weights) {
    const ctx = document.getElementById("canvasWeightsChart");
    if (!ctx || typeof Chart === "undefined") return;
    if (charts.weights) charts.weights.destroy();

    charts.weights = new Chart(ctx, {
      type: "bar",
      data: {
        labels: featureNames,
        datasets: [{
          label: "Weight / Importance",
          data: weights,
          backgroundColor: weights.map((w) => w >= 0 ? "rgba(37, 99, 235, 0.75)" : "rgba(244, 63, 94, 0.75)"),
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false
      }
    });
  }

  return {
    renderBoundaryCanvas,
    renderLossChart,
    renderROCChart,
    renderWeightsChart
  };
})();
