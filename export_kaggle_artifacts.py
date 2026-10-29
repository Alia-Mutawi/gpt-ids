"""Run this cell/script in Kaggle after the existing notebook has saved its artifacts."""
from pathlib import Path
import json
import joblib

ARTIFACT_DIR = Path('/kaggle/working/ids_revised_artifacts')
OUT_DIR = Path('/kaggle/working/ids_dashboard_export')
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load(ARTIFACT_DIR / 'xgb_lightweight_final.pkl')
imputer = joblib.load(ARTIFACT_DIR / 'light_imputer.pkl')
scaler = joblib.load(ARTIFACT_DIR / 'light_scaler.pkl')
features = joblib.load(ARTIFACT_DIR / 'top10_features.pkl')
label_encoder = joblib.load(ARTIFACT_DIR / 'label_encoder.pkl')

# Save XGBoost in its native format rather than shipping the old pickle.
model.save_model(str(OUT_DIR / 'xgb_lightweight_final.json'))
preprocessing = {
    'feature_names': list(features),
    'imputer_statistics': [float(v) for v in imputer.statistics_],
    'scaler_mean': [float(v) for v in scaler.mean_],
    'scaler_scale': [float(v) for v in scaler.scale_],
    'label_classes': [str(v) for v in label_encoder.classes_],
}
with open(OUT_DIR / 'preprocessing.json', 'w', encoding='utf-8') as f:
    json.dump(preprocessing, f, indent=2, ensure_ascii=False)

print('Export complete:', OUT_DIR)
print('Upload xgb_lightweight_final.json and preprocessing.json to the deployment folder.')
