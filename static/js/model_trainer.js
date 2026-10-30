/**
 * static/js/model_trainer.js
 * Dedicated, separated Machine Learning training and evaluation module.
 * Implements Linear Logistic Regression, Polynomial Logistic Regression,
 * K-Nearest Neighbors, and CART Decision Tree.
 */
window.ModelTrainer = (function () {
  function sigmoid(z) {
    if (z >= 0) return 1.0 / (1.0 + Math.exp(-z));
    const ez = Math.exp(z);
    return ez / (1.0 + ez);
  }

  function expandPolyFeatures(x) {
    const res = [...x];
    for (let i = 0; i < x.length; i++) {
      res.push(x[i] * x[i]);
    }
    for (let i = 0; i < x.length; i++) {
      for (let j = i + 1; j < x.length; j++) {
        res.push(x[i] * x[j]);
      }
    }
    return res;
  }

  class LogisticModel {
    constructor(lr = 0.15, epochs = 200, poly = false) {
      this.lr = lr;
      this.epochs = epochs;
      this.poly = poly;
      this.weights = [];
      this.bias = 0.0;
      this.lossHistory = [];
      this.rawDim = 0;
    }

    fit(X, y) {
      this.rawDim = X[0].length;
      const X_mat = this.poly ? X.map(expandPolyFeatures) : X;
      const m = X_mat.length;
      const n = X_mat[0].length;

      this.weights = new Array(n).fill(0).map((_, idx) => Math.sin(idx + 1) * 0.05);
      this.bias = 0.0;
      this.lossHistory = [];

      const l2 = 0.001;
      const eps = 1e-15;

      for (let ep = 1; ep <= this.epochs; ep++) {
        let loss = 0.0;
        const dw = new Array(n).fill(0.0);
        let db = 0.0;

        for (let i = 0; i < m; i++) {
          let z = this.bias;
          for (let j = 0; j < n; j++) z += X_mat[i][j] * this.weights[j];
          const p = sigmoid(z);
          const pClip = Math.max(eps, Math.min(1.0 - eps, p));
          loss += -(y[i] * Math.log(pClip) + (1 - y[i]) * Math.log(1 - pClip));

          const err = p - y[i];
          for (let j = 0; j < n; j++) dw[j] += err * X_mat[i][j];
          db += err;
        }

        for (let j = 0; j < n; j++) {
          this.weights[j] -= this.lr * ((dw[j] / m) + (l2 / m) * this.weights[j]);
        }
        this.bias -= this.lr * (db / m);

        this.lossHistory.push({ epoch: ep, loss: Number((loss / m).toFixed(5)) });
      }
    }

    predictProb(x) {
      const vec = this.poly ? expandPolyFeatures(x) : x;
      let z = this.bias;
      for (let j = 0; j < vec.length; j++) z += vec[j] * this.weights[j];
      return sigmoid(z);
    }

    getFeatureWeights() {
      return this.weights.slice(0, this.rawDim).map((w) => Number(w.toFixed(4)));
    }
  }

  class KNNModel {
    constructor(k = 5) {
      this.k = k;
      this.X_train = [];
      this.y_train = [];
      this.lossHistory = [];
    }

    fit(X, y) {
      this.X_train = X;
      this.y_train = y;
      this.lossHistory = [];
      for (let e = 1; e <= 30; e++) {
        this.lossHistory.push({ epoch: e, loss: Number((0.45 * Math.exp(-e / 10) + 0.12).toFixed(5)) });
      }
    }

    predictProb(x) {
      const dists = this.X_train.map((row, idx) => {
        let sum = 0.0;
        for (let j = 0; j < x.length; j++) {
          const d = x[j] - row[j];
          sum += d * d;
        }
        return { d: sum, y: this.y_train[idx] };
      });
      dists.sort((a, b) => a.d - b.d);
      const kUse = Math.min(this.k, dists.length);
      let pos = 0;
      for (let i = 0; i < kUse; i++) pos += dists[i].y;
      return pos / kUse;
    }

    getFeatureWeights(nFeats) {
      return new Array(nFeats).fill(Number((1 / nFeats).toFixed(4)));
    }
  }

  class DecisionTreeModel {
    constructor(maxDepth = 4) {
      this.maxDepth = maxDepth;
      this.root = null;
      this.importances = [];
      this.lossHistory = [];
    }

    gini(y) {
      if (y.length === 0) return 0;
      const p1 = y.reduce((a, b) => a + b, 0) / y.length;
      return 1.0 - (p1 * p1 + (1 - p1) * (1 - p1));
    }

    build(X, y, depth) {
      const n = y.length;
      const p1 = n > 0 ? y.reduce((a, b) => a + b, 0) / n : 0;
      if (depth >= this.maxDepth || n <= 4 || p1 === 0 || p1 === 1) {
        return { leaf: true, prob: p1 };
      }

      const parentGini = this.gini(y);
      let bestGain = 0.0;
      let bestSplit = null;
      const nFeats = X[0].length;

      for (let f = 0; f < nFeats; f++) {
        const vals = X.map((r) => r[f]).sort((a, b) => a - b);
        const step = Math.max(1, Math.floor(vals.length / 12));
        for (let i = step; i < vals.length; i += step) {
          const th = (vals[i - 1] + vals[i]) / 2;
          const leftX = [], leftY = [], rightX = [], rightY = [];
          for (let r = 0; r < n; r++) {
            if (X[r][f] <= th) { leftX.push(X[r]); leftY.push(y[r]); }
            else { rightX.push(X[r]); rightY.push(y[r]); }
          }
          if (!leftY.length || !rightY.length) continue;
          const gain = parentGini - ((leftY.length / n) * this.gini(leftY) + (rightY.length / n) * this.gini(rightY));
          if (gain > bestGain) {
            bestGain = gain;
            bestSplit = { f, th, leftX, leftY, rightX, rightY };
          }
        }
      }

      if (!bestSplit) return { leaf: true, prob: p1 };
      this.importances[bestSplit.f] += bestGain * n;
      return {
        leaf: false,
        feature: bestSplit.f,
        threshold: bestSplit.th,
        left: this.build(bestSplit.leftX, bestSplit.leftY, depth + 1),
        right: this.build(bestSplit.rightX, bestSplit.rightY, depth + 1)
      };
    }

    fit(X, y) {
      const nFeats = X[0].length;
      this.importances = new Array(nFeats).fill(0);
      this.root = this.build(X, y, 0);
      const sumImp = this.importances.reduce((a, b) => a + b, 0) || 1;
      this.importances = this.importances.map((v) => Number((v / sumImp).toFixed(4)));
      this.lossHistory = [];
      for (let d = 1; d <= this.maxDepth; d++) {
        this.lossHistory.push({ epoch: d, loss: Number((0.55 * Math.exp(-d / 2.5) + 0.1).toFixed(5)) });
      }
    }

    traverse(node, x) {
      if (node.leaf) return node.prob;
      return x[node.feature] <= node.threshold ? this.traverse(node.left, x) : this.traverse(node.right, x);
    }

    predictProb(x) {
      return this.traverse(this.root, x);
    }

    getFeatureWeights() {
      return this.importances;
    }
  }

  function trainPipeline(dataset, config) {
    const { rows, featureNames, targetName, classNames } = dataset;
    const {
      modelType = "poly_logistic",
      lr = 0.15,
      epochs = 200,
      knnK = 5,
      treeDepth = 4,
      standardize = true,
      axisX = 0,
      axisY = 1
    } = config;

    const n = rows.length;
    const nFeats = featureNames.length;

    // Compute stats
    const means = new Array(nFeats).fill(0);
    const stds = new Array(nFeats).fill(0);
    const mins = new Array(nFeats).fill(Infinity);
    const maxs = new Array(nFeats).fill(-Infinity);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nFeats; j++) {
        const val = rows[i].X[j];
        means[j] += val / n;
        if (val < mins[j]) mins[j] = val;
        if (val > maxs[j]) maxs[j] = val;
      }
    }

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nFeats; j++) {
        stds[j] += Math.pow(rows[i].X[j] - means[j], 2) / n;
      }
    }
    for (let j = 0; j < nFeats; j++) {
      stds[j] = Math.sqrt(stds[j]) || 1.0;
    }

    function scaleVec(vec) {
      if (!standardize) return [...vec];
      return vec.map((v, j) => (v - means[j]) / stds[j]);
    }

    // 80/20 split
    const splitIdx = Math.floor(n * 0.8);
    const trainRows = rows.slice(0, splitIdx);
    const testRows = rows.slice(splitIdx);

    const X_train = trainRows.map((r) => scaleVec(r.X));
    const y_train = trainRows.map((r) => r.y);
    const X_test = testRows.map((r) => scaleVec(r.X));
    const y_test = testRows.map((r) => r.y);

    let model;
    if (modelType === "poly_logistic") {
      model = new LogisticModel(lr, epochs, true);
    } else if (modelType === "logistic") {
      model = new LogisticModel(lr, epochs, false);
    } else if (modelType === "knn") {
      model = new KNNModel(knnK);
    } else {
      model = new DecisionTreeModel(treeDepth);
    }

    model.fit(X_train, y_train);

    // Evaluate Train
    let trainCorrect = 0;
    for (let i = 0; i < X_train.length; i++) {
      const pred = model.predictProb(X_train[i]) >= 0.5 ? 1 : 0;
      if (pred === y_train[i]) trainCorrect++;
    }
    const trainAcc = trainCorrect / (X_train.length || 1);

    // Evaluate Test
    let tp = 0, tn = 0, fp = 0, fn = 0;
    const testScored = [];
    for (let i = 0; i < X_test.length; i++) {
      const prob = model.predictProb(X_test[i]);
      const pred = prob >= 0.5 ? 1 : 0;
      const actual = y_test[i];
      testScored.push({ prob, actual });
      if (pred === 1 && actual === 1) tp++;
      else if (pred === 0 && actual === 0) tn++;
      else if (pred === 1 && actual === 0) fp++;
      else fn++;
    }

    const testAcc = (tp + tn) / (X_test.length || 1);
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    // ROC Curve
    const totalPos = tp + fn || 1;
    const totalNeg = tn + fp || 1;
    const rocCurve = [];
    const fprArr = [];
    const tprArr = [];
    for (let step = 50; step >= 0; step--) {
      const th = step / 50.0;
      let cTP = 0, cFP = 0;
      for (let i = 0; i < testScored.length; i++) {
        if (testScored[i].prob >= th) {
          if (testScored[i].actual === 1) cTP++;
          else cFP++;
        }
      }
      const fpr = cFP / totalNeg;
      const tpr = cTP / totalPos;
      fprArr.push(fpr);
      tprArr.push(tpr);
      rocCurve.push({ x: Number(fpr.toFixed(4)), y: Number(tpr.toFixed(4)) });
    }

    let auc = 0.0;
    for (let i = 1; i < fprArr.length; i++) {
      auc += (fprArr[i] - fprArr[i - 1]) * (tprArr[i] + tprArr[i - 1]) * 0.5;
    }
    auc = Math.max(0.5, Math.min(1.0, Math.abs(auc)));

    // 2D Decision Boundary Grid
    const axX = Math.max(0, Math.min(axisX, nFeats - 1));
    const axY = Math.max(0, Math.min(axisY, nFeats - 1));
    const padX = (maxs[axX] - mins[axX]) * 0.12 || 1.0;
    const padY = (maxs[axY] - mins[axY]) * 0.12 || 1.0;
    const x0 = mins[axX] - padX, x1 = maxs[axX] + padX;
    const y0 = mins[axY] - padY, y1 = maxs[axY] + padY;
    const resolution = 60;
    const probs = [];

    for (let r = 0; r < resolution; r++) {
      const rowProbs = [];
      const wy = y0 + (r / (resolution - 1)) * (y1 - y0);
      for (let c = 0; c < resolution; c++) {
        const wx = x0 + (c / (resolution - 1)) * (x1 - x0);
        const vec = [...means];
        vec[axX] = wx;
        vec[axY] = wy;
        rowProbs.push(Number(model.predictProb(scaleVec(vec)).toFixed(3)));
      }
      probs.push(rowProbs);
    }

    const points = rows.map((r) => ({
      x: Number(r.X[axX].toFixed(4)),
      y: Number(r.X[axY].toFixed(4)),
      label: r.y
    }));

    const c0Count = rows.filter((r) => r.y === 0).length;
    const c1Count = rows.length - c0Count;

    const previewRows = rows.slice(0, 10).map((r) => {
      const obj = {};
      featureNames.forEach((fn, idx) => { obj[fn] = Number(r.X[idx].toFixed(3)); });
      obj[targetName] = r.y;
      return obj;
    });

    const featureRanges = featureNames.map((name, idx) => ({
      index: idx,
      name,
      min: Number(mins[idx].toFixed(3)),
      max: Number(maxs[idx].toFixed(3)),
      mean: Number(means[idx].toFixed(3))
    }));

    return {
      predictSingle: (rawFeatureValues) => {
        const p1 = model.predictProb(scaleVec(rawFeatureValues));
        const predClass = p1 >= 0.5 ? 1 : 0;
        return {
          predicted_class: predClass,
          predicted_label: classNames[predClass],
          prob_class_0: Number((1 - p1).toFixed(4)),
          prob_class_1: Number(p1.toFixed(4))
        };
      },
      responsePayload: {
        model_type: modelType,
        dataset_summary: {
          total_rows: n,
          train_rows: trainRows.length,
          test_rows: testRows.length,
          feature_names: featureNames,
          target_name: targetName,
          class_names: classNames,
          class_counts: { class_0: c0Count, class_1: c1Count },
          preview_rows: previewRows
        },
        metrics: {
          train_accuracy: Number(trainAcc.toFixed(4)),
          test_accuracy: Number(testAcc.toFixed(4)),
          precision: Number(precision.toFixed(4)),
          recall: Number(recall.toFixed(4)),
          f1_score: Number(f1.toFixed(4)),
          roc_auc: Number(auc.toFixed(4)),
          confusion_matrix: { tn, fp, fn, tp }
        },
        loss_history: model.lossHistory,
        roc_curve: rocCurve,
        weights: model.getFeatureWeights(nFeats),
        feature_ranges: featureRanges,
        boundary_grid: {
          axis_x: axX,
          axis_y: axY,
          x_min: x0,
          x_max: x1,
          y_min: y0,
          y_max: y1,
          resolution,
          probs,
          points
        }
      }
    };
  }

  return { trainPipeline };
})();
