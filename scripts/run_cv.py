import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedKFold
from src.classifier.rules import StabilityRuleEngine
from src.classifier.models import StabilityLabel, StabilityResult, ClassificationSource
from src.classifier.fallback import BaseFallbackClassifier, FallbackPrediction
from src.evaluation.stability_evaluator import StabilityEvaluator
import joblib
from scripts.train_fallback import build_pipeline
def evaluate_2stage_cv():
    evaluator = StabilityEvaluator()
    bench_path = ROOT / "data" / "raw" / "query_stability_benchmark.json"
    human_path = ROOT / "data" / "raw" / "query_stability_human_credibility.json"
    
    bench_items = evaluator.load_benchmark_dataset(bench_path)
    human_items = evaluator.load_benchmark_dataset(human_path)
    
    all_items = bench_items + human_items
    texts = [item["query"] for item in all_items]
    y_true = np.array([1 if item["stability_label"] == "DYNAMIC" else 0 for item in all_items])
    domains = [item.get("domain", "general") for item in all_items]
    
    rule_engine = StabilityRuleEngine()
    
    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_predictions = []
    cv_sources = []
    cv_confidences = []
    cv_rationales = []
    
    # Pre-allocate
    pred_labels = np.zeros(len(texts), dtype=int)
    pred_sources = [""] * len(texts)
    pred_confs = np.zeros(len(texts), dtype=float)
    pred_rules = [None] * len(texts)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y_true)):
        # Train fallback on train_idx
        X_train = [texts[i] for i in train_idx]
        y_train = y_true[train_idx]
        
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        
        # Evaluate val_idx
        for i in val_idx:
            q = texts[i]
            rule_match = rule_engine.evaluate(q)
            if rule_match.matched:
                pred_labels[i] = 1 if rule_match.label == StabilityLabel.DYNAMIC else 0
                pred_sources[i] = "RULE"
                pred_confs[i] = rule_match.confidence
                pred_rules[i] = rule_match.rule_name
            else:
                # Run trained fallback on held-out sample
                proba = pipeline.predict_proba([q])[0]
                p_dynamic = proba[1]
                
                # Default logic: if p_dynamic >= 0.50 -> DYNAMIC, else STABLE
                if p_dynamic >= 0.50:
                    pred_labels[i] = 1
                    pred_confs[i] = p_dynamic
                else:
                    conf = 1.0 - p_dynamic
                    # Sub-stage A threshold check: if STABLE with conf < 0.80 -> override to DYNAMIC
                    if conf < 0.80:
                        pred_labels[i] = 1  # DYNAMIC override
                        pred_confs[i] = conf
                    else:
                        pred_labels[i] = 0
                        pred_confs[i] = conf
                        
                pred_sources[i] = "FALLBACK"
                pred_rules[i] = None
                
    total = len(y_true)
    correct = np.sum(pred_labels == y_true)
    acc = correct / total * 100
    
    dyn_mask = (y_true == 1)
    stab_mask = (y_true == 0)
    
    fp_danger = np.sum((y_true == 1) & (pred_labels == 0)) # Dynamic called Stable
    fn_conserv = np.sum((y_true == 0) & (pred_labels == 1)) # Stable called Dynamic
    
    danger_rate = fp_danger / np.sum(dyn_mask) * 100
    conserv_rate = fn_conserv / np.sum(stab_mask) * 100
    
    rule_resolved = sum(1 for s in pred_sources if s == "RULE")
    fallback_resolved = sum(1 for s in pred_sources if s == "FALLBACK")
    
    print("\n" + "="*70)
    print("5-FOLD STRATIFIED CV RESULTS (COMPLETE TWO-STAGE PIPELINE)")
    print("="*70)
    print(f"Total Corpus Queries       : {total}")
    print(f"Stratified CV Accuracy     : {acc:.2f}% ({correct}/{total})")
    print(f"CV Dangerous Errors (FP)   : {danger_rate:.2f}% ({fp_danger}/{np.sum(dyn_mask)})")
    print(f"CV Conservative Errors (FN): {conserv_rate:.2f}% ({fn_conserv}/{np.sum(stab_mask)})")
    print(f"Pipeline Resolution Split  : Stage 1 Rules={rule_resolved} ({rule_resolved/total*100:.1f}%), Stage 2 Fallback={fallback_resolved} ({fallback_resolved/total*100:.1f}%)")
    
    # Check separate datasets under CV
    n_bench = len(bench_items)
    print("\n--- CV Breakdown: 160-Query Dev Benchmark ---")
    b_true = y_true[:n_bench]
    b_pred = pred_labels[:n_bench]
    b_danger = np.sum((b_true == 1) & (b_pred == 0))
    b_conserv = np.sum((b_true == 0) & (b_pred == 1))
    print(f"  Accuracy: {np.mean(b_true == b_pred)*100:.2f}% | Danger: {b_danger}/{np.sum(b_true==1)} | Conserv: {b_conserv}/{np.sum(b_true==0)}")
    
    print("\n--- CV Breakdown: 85-Query Human Credibility Audit ---")
    h_true = y_true[n_bench:]
    h_pred = pred_labels[n_bench:]
    h_danger = np.sum((h_true == 1) & (h_pred == 0))
    h_conserv = np.sum((h_true == 0) & (h_pred == 1))
    print(f"  Accuracy: {np.mean(h_true == h_pred)*100:.2f}% | Danger: {h_danger}/{np.sum(h_true==1)} ({h_danger/np.sum(h_true==1)*100:.2f}%) | Conserv: {h_conserv}/{np.sum(h_true==0)} ({h_conserv/np.sum(h_true==0)*100:.2f}%)")
    
    if h_danger > 0:
        print("\n  Remaining Dangerous Errors on 85-set in CV:")
        for idx in range(n_bench, total):
            if y_true[idx] == 1 and pred_labels[idx] == 0:
                print(f"    [{all_items[idx]['id']}] ({pred_sources[idx]} | conf={pred_confs[idx]:.2f}): '{texts[idx]}'")

if __name__ == "__main__":
    evaluate_2stage_cv()
