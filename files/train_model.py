"""
train_model.py
==============
Run this ONCE before launching the Streamlit app.
Generates: best_churn_model.pkl, scaler.pkl, model_meta.pkl

Usage:
    python train_model.py

Place European_Bank.csv in the same directory before running.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve)
import joblib, warnings
warnings.filterwarnings('ignore')
import pickle
import joblib

# ── 1. Load & clean ───────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('European_Bank.csv')
df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

# ── 2. Feature engineering ────────────────────────────────────
print("⚙️  Engineering features...")
df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['ProductDensity']     = df['NumOfProducts'] / (df['Tenure'] + 1)
df['EngagementProduct']  = df['IsActiveMember'] * df['NumOfProducts']
df['AgeTenure']          = df['Age'] * df['Tenure']
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

X = df.drop('Exited', axis=1)
y = df['Exited']
feature_cols = X.columns.tolist()
print(f"   Features: {feature_cols}")
print(f"   Shape: {X.shape}  |  Churn rate: {y.mean():.2%}")

# ── 3. Split & scale ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Oversample minority class ─────────────────────────────
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]
extra   = np.random.RandomState(42).choice(pos_idx, size=len(neg_idx)-len(pos_idx), replace=True)
idx_all = np.concatenate([np.arange(len(y_train)), extra])
X_bal   = X_train_scaled[idx_all]
y_bal   = np.concatenate([y_train.values, y_train.values[extra]])
print(f"   Balanced training set: {np.sum(y_bal==0)} neg / {np.sum(y_bal==1)} pos")

# ── 5. Train models ───────────────────────────────────────────
print("\n🤖 Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                                   min_samples_split=20, max_features='sqrt',
                                                   random_state=42, class_weight='balanced'),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                       learning_rate=0.05, subsample=0.8, random_state=42),
}

# Try XGBoost
try:
    from xgboost import XGBClassifier
    models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       reg_alpha=0.1, reg_lambda=1.5,
                                       random_state=42, eval_metric='logloss',
                                       use_label_encoder=False)
    print("   XGBoost available ✅")
except ImportError:
    print("   XGBoost not installed — skipping")

results = {}
for name, model in models.items():
    print(f"   Training {name}...", end=" ", flush=True)
    model.fit(X_bal, y_bal)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'model':     model,
        'accuracy':  round(accuracy_score(y_test, y_pred)  * 100, 2),
        'precision': round(precision_score(y_test, y_pred) * 100, 2),
        'recall':    round(recall_score(y_test, y_pred)    * 100, 2),
        'f1':        round(f1_score(y_test, y_pred)        * 100, 2),
        'roc_auc':   round(roc_auc_score(y_test, y_prob)   * 100, 2),
    }
    print(f"Acc={results[name]['accuracy']}%  AUC={results[name]['roc_auc']}%  F1={results[name]['f1']}%")

# ── 6. Select best model ──────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['roc_auc'])
best_model = results[best_name]['model']
print(f"\n🏆 Best model: {best_name}")

# ── 7. Threshold tuning ───────────────────────────────────────
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
prec, rec, thresholds = precision_recall_curve(y_test, y_prob_best)
f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
best_idx       = f1s.argmax()
best_threshold = float(thresholds[best_idx])
print(f"   Best threshold (max-F1): {best_threshold:.4f}")
print(f"   Precision: {prec[best_idx]:.4f}  Recall: {rec[best_idx]:.4f}  F1: {f1s[best_idx]:.4f}")

# ── 8. Feature importance (Random Forest) ────────────────────
rf_model  = results['Random Forest']['model']
feat_imp  = dict(zip(feature_cols, rf_model.feature_importances_))
feat_imp  = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

# ── 9. Save ───────────────────────────────────────────────────
print("\n💾 Saving artifacts...")
joblib.dump(best_model, 'best_churn_model.pkl')
joblib.dump(scaler,     'scaler.pkl')
joblib.dump({
    'results':           {k: {m: v for m, v in r.items() if m != 'model'} for k, r in results.items()},
    'all_models':        {k: r['model'] for k, r in results.items()},
    'best_name':         best_name,
    'best_threshold':    best_threshold,
    'feature_cols':      feature_cols,
    'feature_importance': feat_imp,
}, 'model_meta.pkl')

print("✅ Done! Files saved:")
print("   • best_churn_model.pkl")
print("   • scaler.pkl")
print("   • model_meta.pkl")
print("\nNow launch the app:")
print("   streamlit run app.py")
