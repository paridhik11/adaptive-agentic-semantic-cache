import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"c:\Users\parid\Downloads\Agentic AI\adaptive-agentic-semantic-cache")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

from src.evaluation.stability_evaluator import StabilityEvaluator

def load_training_corpus():
    evaluator = StabilityEvaluator()
    bench_path = ROOT / "data" / "raw" / "query_stability_benchmark.json"
    human_path = ROOT / "data" / "raw" / "query_stability_human_credibility.json"
    
    bench_items = evaluator.load_benchmark_dataset(bench_path)
    human_items = evaluator.load_benchmark_dataset(human_path)
    
    all_items = bench_items + human_items
    print(f"Loaded {len(bench_items)} benchmark + {len(human_items)} human = {len(all_items)} total queries.")
    
    texts = [item["query"] for item in all_items]
    labels = [1 if item["stability_label"] == "DYNAMIC" else 0 for item in all_items] # 1 = DYNAMIC, 0 = STABLE
    return texts, labels, all_items

def build_pipeline():
    union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("char_tfidf", TfidfVectorizer(
            ngram_range=(2, 5),
            analyzer="char_wb",
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
    ])
    
    clf = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    
    return Pipeline([
        ("features", union),
        ("classifier", clf)
    ])

def main():
    texts, labels, items = load_training_corpus()
    X = texts
    y = np.array(labels)
    
    pipeline = build_pipeline()
    
    # 5-fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")
    
    cv_preds = (cv_probs[:, 1] >= 0.50).astype(int)
    
    acc = accuracy_score(y, cv_preds)
    cm = confusion_matrix(y, cv_preds) # [[TN (Stable), FP (Danger)], [FN (Conserv), TP (Dynamic)]]
    # Note: in sklearn confusion matrix with y in {0,1}:
    # 0 = STABLE, 1 = DYNAMIC
    # cm[0,0] = true stable (predicted stable) -> TP in cache terms
    # cm[0,1] = stable predicted dynamic (conservative error) -> FN in cache terms
    # cm[1,0] = dynamic predicted stable (dangerous error) -> FP in cache terms
    # cm[1,1] = dynamic predicted dynamic (true dynamic) -> TN in cache terms
    
    tn_stable = cm[0, 0]
    fp_conserv = cm[0, 1]
    fn_danger = cm[1, 0]
    tp_dynamic = cm[1, 1]
    
    total_dynamic = sum(y == 1)
    total_stable = sum(y == 0)
    
    print("\n" + "="*70)
    print("5-FOLD STRATIFIED CROSS-VALIDATION RESULTS (STANDALONE FALLBACK)")
    print("="*70)
    print(f"Total CV queries        : {len(y)}")
    print(f"CV Accuracy             : {acc*100:.2f}% ({sum(y == cv_preds)}/{len(y)})")
    print(f"CV Dangerous Errors (FP): {fn_danger}/{total_dynamic} ({fn_danger/total_dynamic*100:.2f}%)")
    print(f"CV Conservative Errors  : {fp_conserv}/{total_stable} ({fp_conserv/total_stable*100:.2f}%)")
    
    # Calibration Check
    print("\n" + "-"*70)
    print("PROVISIONAL CALIBRATION CHECK (CV Predicted Probabilities for DYNAMIC)")
    print("-"*70)
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    dyn_probs = cv_probs[:, 1]
    for low, high in buckets:
        mask = (dyn_probs >= low) & (dyn_probs < high) if high < 1.0 else (dyn_probs >= low) & (dyn_probs <= high)
        n_bucket = np.sum(mask)
        if n_bucket > 0:
            mean_pred = np.mean(dyn_probs[mask])
            emp_dynamic_rate = np.mean(y[mask])
            print(f"Bucket [{low:.1f}, {high:.1f}]: count={n_bucket:3d} | Mean Pred P(Dyn)={mean_pred*100:5.1f}% | Empirical Dyn={emp_dynamic_rate*100:5.1f}%")
        else:
            print(f"Bucket [{low:.1f}, {high:.1f}]: count=  0")
            
    # Train final artifact on full 245 dataset
    pipeline.fit(X, y)
    
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "stability_fallback_lexical.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nFinal model artifact saved to: {model_path}")
    
if __name__ == "__main__":
    main()
