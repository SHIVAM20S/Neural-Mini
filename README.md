# ML Classification Dashboard (Modular Project)

A modular Machine Learning project with separated dataset generation, model training, visualization, and UI layers. Works both by opening `index.html` directly in any browser (or VS Code Live Server) AND by running the Python backend (`python app.py` or `python train_cli.py`).

## Project Structure

```text
ml_project/
├── index.html                    # Main dashboard HTML (clean markup, no inline JS/CSS)
├── app.py                        # Python backend server
├── train_cli.py                  # Standalone CLI script to train & save models (.pkl)
├── requirements.txt              # Python dependencies (numpy, pandas)
├── data/
│   ├── generator.py              # Python dataset generator & CSV loader
│   └── sample_datasets/          # Sample CSV files (Churn, Iris, Exam Pass)
├── models/
│   ├── trainer.py                # Python ML trainer (Logistic, Poly Logistic, KNN, Tree)
│   └── saved/                    # Serialized trained_model.pkl & latest_metrics.json
└── static/
    ├── css/styles.css            # Stylesheet
    └── js/
        ├── data_generator.js     # Separated dataset generator & CSV parser module
        ├── model_trainer.js      # Separated ML model training module
        ├── visualizer.js         # Separated Canvas & Chart.js plotting module
        └── dashboard.js          # Main UI controller
```

## How to Run

1. **Instant Browser / Live Server Mode**:
   - Open `index.html` directly in your browser (double-click) or with VS Code Live Server.
   - It loads the separated modules (`static/js/data_generator.js`, `static/js/model_trainer.js`, `static/js/visualizer.js`, `static/js/dashboard.js`) and renders all visualizations immediately.

2. **Python Backend Server Mode**:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
   Then open `http://127.0.0.1:5000`.

3. **Standalone Python CLI Training**:
   ```bash
   python train_cli.py --preset moons --samples 240 --noise 0.18 --model poly_logistic
   ```
