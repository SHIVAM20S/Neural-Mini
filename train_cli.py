#!/usr/bin/env python3
"""
train_cli.py
Standalone command-line script to train and evaluate Machine Learning models
completely separately from the web dashboard UI.

Usage Examples:
  1. Train Logistic Regression on synthetic Two Moons dataset:
     python3 train_cli.py --preset moons --samples 240 --noise 0.18 --model poly_logistic --lr 0.15 --epochs 200

  2. Train Decision Tree on a custom CSV dataset:
     python3 train_cli.py --csv data/sample_datasets/customer_churn.csv --target Churn_Status --model decision_tree
"""
import argparse
import json
import os
from data.generator import generate_synthetic_data, load_and_preprocess_csv
from models.trainer import MLModelPipeline


def main():
    parser = argparse.ArgumentParser(description="Standalone ML Model Trainer CLI")
    parser.add_argument("--csv", type=str, default=None, help="Path to custom CSV dataset file")
    parser.add_argument("--target", type=str, default=None, help="Target column name (for CSV)")
    parser.add_argument("--preset", type=str, default="moons", choices=["moons", "circles", "blobs", "xor", "churn"], help="Synthetic dataset preset")
    parser.add_argument("--samples", type=int, default=240, help="Number of synthetic samples")
    parser.add_argument("--noise", type=float, default=0.18, help="Synthetic noise level")
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "poly_logistic", "knn", "decision_tree"], help="ML model architecture")
    parser.add_argument("--lr", type=float, default=0.15, help="Learning rate for Logistic Regression")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs for Logistic Regression")
    parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization strength")
    parser.add_argument("--k", type=int, default=5, help="Neighbors K for KNN")
    parser.add_argument("--depth", type=int, default=4, help="Max depth for Decision Tree")
    parser.add_argument("--output-json", type=str, default="models/saved/latest_metrics.json", help="Path to save evaluation metrics JSON")

    args = parser.parse_args()

    print("=" * 68)
    print("  STANDALONE MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 68)

    if args.csv:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        with open(args.csv, "r", encoding="utf-8") as f:
            csv_text = f.read()
        df, feature_names, target_name, class_names = load_and_preprocess_csv(csv_text, target_col=args.target)
        print(f"[DATA] Loaded CSV '{args.csv}' -> {len(df)} rows, Features: {feature_names}, Target: '{target_name}'")
    else:
        df, feature_names, target_name, class_names = generate_synthetic_data(
            preset=args.preset, n_samples=args.samples, noise=args.noise
        )
        print(f"[DATA] Generated Synthetic '{args.preset}' -> {len(df)} rows, Features: {feature_names}")

    pipeline = MLModelPipeline(save_dir="models/saved")
    hyperparams = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "l2_lambda": args.l2,
        "n_neighbors": args.k,
        "max_depth": args.depth,
        "threshold": 0.5
    }

    results = pipeline.train_and_evaluate(
        df=df,
        feature_names=feature_names,
        target_name=target_name,
        class_names=class_names,
        model_type=args.model,
        hyperparams=hyperparams,
        standardize=True
    )

    m = results["metrics"]
    cm = m["confusion_matrix"]

    print("-" * 68)
    print(f"[MODEL] Architecture       : {args.model.upper()}")
    print(f"[METRIC] Train Accuracy    : {m['train_accuracy'] * 100:.2f}%")
    print(f"[METRIC] Test Accuracy     : {m['test_accuracy'] * 100:.2f}%")
    print(f"[METRIC] Precision         : {m['precision']:.4f}")
    print(f"[METRIC] Recall            : {m['recall']:.4f}")
    print(f"[METRIC] F1-Score          : {m['f1_score']:.4f}")
    print(f"[METRIC] ROC AUC           : {m['roc_auc']:.4f}")
    print(f"[MATRIX] Confusion Matrix  : TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    print(f"[WEIGHTS] Feature Weights  : {dict(zip(feature_names, results['weights']))}")
    print("-" * 68)
    print(f"[SAVED] Serialized model   -> {results['artifact_path']}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f_json:
        summary_out = {k: v for k, v in results.items() if k != "boundary_grid"}
        json.dump(summary_out, f_json, indent=2)
    print(f"[SAVED] Evaluation metrics -> {args.output_json}")
    print("=" * 68)


if __name__ == "__main__":
    main()
