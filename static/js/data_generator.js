/**
 * static/js/data_generator.js
 * Separated module for synthetic dataset generation, benchmark datasets, and CSV parsing.
 */
window.DataGenerator = (function () {
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function generateSynthetic(preset, nSamples, noise) {
    const rows = [];
    let featureNames = ["X1", "X2"];
    let targetName = "Target";
    let classNames = ["Class 0", "Class 1"];
    const nHalf = Math.floor(nSamples / 2);

    if (preset === "moons") {
      featureNames = ["Curvature_X", "Curvature_Y"];
      classNames = ["Moon A (0)", "Moon B (1)"];
      for (let i = 0; i < nHalf; i++) {
        const t = Math.PI * (i / nHalf);
        rows.push({ X: [Math.cos(t) + noise * randn(), Math.sin(t) + noise * randn()], y: 0 });
      }
      for (let i = 0; i < nSamples - nHalf; i++) {
        const t = Math.PI * (i / (nSamples - nHalf));
        rows.push({ X: [1 - Math.cos(t) + noise * randn(), 0.5 - Math.sin(t) + noise * randn()], y: 1 });
      }
    } else if (preset === "circles") {
      featureNames = ["Radius_X", "Radius_Y"];
      classNames = ["Outer Ring (0)", "Inner Core (1)"];
      for (let i = 0; i < nHalf; i++) {
        const t = Math.random() * 2 * Math.PI;
        const r = 1.0 + noise * randn();
        rows.push({ X: [r * Math.cos(t), r * Math.sin(t)], y: 0 });
      }
      for (let i = 0; i < nSamples - nHalf; i++) {
        const t = Math.random() * 2 * Math.PI;
        const r = 0.42 + noise * 0.8 * randn();
        rows.push({ X: [r * Math.cos(t), r * Math.sin(t)], y: 1 });
      }
    } else if (preset === "blobs") {
      featureNames = ["Dimension_1", "Dimension_2"];
      classNames = ["Cluster 0", "Cluster 1"];
      const spread = 0.65 + noise * 2.5;
      for (let i = 0; i < nHalf; i++) {
        rows.push({ X: [-1.4 + spread * randn(), -1.2 + spread * randn()], y: 0 });
      }
      for (let i = 0; i < nSamples - nHalf; i++) {
        rows.push({ X: [1.4 + spread * randn(), 1.2 + spread * randn()], y: 1 });
      }
    } else if (preset === "xor") {
      featureNames = ["Coord_U", "Coord_V"];
      classNames = ["Even Quadrant (0)", "Odd Quadrant (1)"];
      for (let i = 0; i < nSamples; i++) {
        const u = (Math.random() * 4 - 2);
        const v = (Math.random() * 4 - 2);
        const label = (u * v > 0) ? 1 : 0;
        rows.push({ X: [u + noise * randn(), v + noise * randn()], y: label });
      }
    } else if (preset === "churn") {
      featureNames = ["Customer_Age", "Monthly_Usage_Hrs", "Support_Tickets"];
      targetName = "Churn_Status";
      classNames = ["Retained (0)", "Churned (1)"];
      for (let i = 0; i < nSamples; i++) {
        const age = 20 + Math.random() * 50 + noise * 10 * randn();
        const usage = 5 + Math.random() * 80 + noise * 15 * randn();
        const tickets = Math.max(0, Math.round(Math.random() * 7 + noise * 2 * randn()));
        const logit = -3.0 + 0.045 * age - 0.04 * usage + 0.75 * tickets + noise * 2 * randn();
        const p = 1 / (1 + Math.exp(-logit));
        rows.push({ X: [age, usage, tickets], y: p >= 0.5 ? 1 : 0 });
      }
    }

    return { rows, featureNames, targetName, classNames };
  }

  function getSampleDataset(name) {
    if (name === "churn") {
      const csv = `Customer_Age,Monthly_Usage_Hrs,Support_Tickets,Churn_Status
24,48.5,1,0
52,12.0,5,1
35,62.0,0,0
61,9.5,4,1
29,40.0,1,0
58,15.2,4,1
41,34.0,2,0
49,11.5,5,1
23,70.0,0,0
37,51.0,1,0
55,14.0,3,1
31,44.5,1,0
64,8.0,6,1
27,55.0,0,0
46,18.0,3,1
33,49.0,1,0
59,10.0,5,1
26,60.5,0,0
44,22.0,3,1
39,38.0,2,0`;
      return parseCSVToDataset(csv);
    } else if (name === "iris") {
      const csv = `Sepal_Length,Sepal_Width,Petal_Length,Petal_Width,Species
5.1,3.5,1.4,0.2,Setosa
4.9,3.0,1.4,0.2,Setosa
4.7,3.2,1.3,0.2,Setosa
4.6,3.1,1.5,0.2,Setosa
5.0,3.6,1.4,0.2,Setosa
5.4,3.9,1.7,0.4,Setosa
4.6,3.4,1.4,0.3,Setosa
5.0,3.4,1.5,0.2,Setosa
4.4,2.9,1.4,0.2,Setosa
4.9,3.1,1.5,0.1,Setosa
7.0,3.2,4.7,1.4,Versicolor
6.4,3.2,4.5,1.5,Versicolor
6.9,3.1,4.9,1.5,Versicolor
5.5,2.3,4.0,1.3,Versicolor
6.5,2.8,4.6,1.5,Versicolor
5.7,2.8,4.5,1.3,Versicolor
6.3,3.3,4.7,1.6,Versicolor
4.9,2.4,3.3,1.0,Versicolor
6.6,2.9,4.6,1.3,Versicolor
5.2,2.7,3.9,1.4,Versicolor`;
      return parseCSVToDataset(csv);
    } else {
      const csv = `Study_Hours,Practice_Exams,Passed
1.5,1,0
2.0,1,0
2.5,2,0
3.0,2,0
3.5,3,0
4.2,3,0
4.8,4,1
5.2,5,1
5.8,5,1
6.5,6,1
7.0,7,1
7.8,8,1
8.5,8,1
9.0,9,1`;
      return parseCSVToDataset(csv);
    }
  }

  function parseRawCSV(csvText) {
    const lines = csvText.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length < 2) throw new Error("CSV must have a header and at least one data row.");
    const delim = (lines[0].includes(";") && !lines[0].includes(",")) ? ";" : ",";
    const headers = lines[0].split(delim).map((h) => h.trim().replace(/^"|"$/g, ""));
    const rawRows = lines.slice(1).map((line) => line.split(delim).map((c) => c.trim().replace(/^"|"$/g, "")));
    return { headers, rawRows };
  }

  function parseCSVToDataset(csvText, targetColName = null) {
    const { headers, rawRows } = parseRawCSV(csvText);
    let targetIdx = headers.length - 1;
    if (targetColName && headers.indexOf(targetColName) !== -1) {
      targetIdx = headers.indexOf(targetColName);
    }
    const featIndices = headers.map((_, i) => i).filter((i) => i !== targetIdx);
    const featureNames = featIndices.map((i) => headers[i]);
    const targetName = headers[targetIdx];

    const rawTargets = rawRows.map((r) => r[targetIdx]);
    const uniqueTargets = Array.from(new Set(rawTargets)).sort();
    const posVal = uniqueTargets.length >= 2 ? uniqueTargets[1] : "1";
    const classNames = uniqueTargets.length >= 2
      ? [`${uniqueTargets[0]} (0)`, `${uniqueTargets[1]} (1)`]
      : ["Class 0", "Class 1"];

    const rows = rawRows.map((r) => {
      const X = featIndices.map((fi) => {
        const v = parseFloat(r[fi]);
        return isNaN(v) ? 0.0 : v;
      });
      const y = (r[targetIdx] === posVal || r[targetIdx] === "1") ? 1 : 0;
      return { X, y };
    });

    return { rows, featureNames, targetName, classNames, headers };
  }

  return {
    generateSynthetic,
    getSampleDataset,
    parseRawCSV,
    parseCSVToDataset
  };
})();
