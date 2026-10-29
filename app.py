from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="IoT IDS | Botnet & Slowloris",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1400px; padding-top: 1.4rem; direction: rtl;}
      h1, h2, h3, p, label, div[data-testid="stMetricLabel"] {text-align: right;}
      div[data-testid="stDataFrame"] {direction: ltr;}
      .small-note {font-size: 0.92rem; color: #5f6368;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load native XGBoost model plus numeric preprocessing parameters."""
    with open(BASE_DIR / "preprocessing.json", "r", encoding="utf-8") as f:
        prep = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(str(BASE_DIR / "xgb_lightweight_final.json"))

    features = prep["feature_names"]
    n_features = len(features)
    for key in ("imputer_statistics", "scaler_mean", "scaler_scale"):
        if len(prep[key]) != n_features:
            raise ValueError(f"Invalid preprocessing file: {key} length does not match feature count.")
    if int(model.get_booster().num_features()) != n_features:
        raise ValueError("The loaded model feature count does not match preprocessing.json.")
    return model, prep


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match the exact column normalization used in the training notebook."""
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )
    return result


def predict_flows(raw_df: pd.DataFrame, model, prep):
    """Predict from precomputed flow-feature rows; does not capture packets."""
    normalized = normalize_columns(raw_df)
    features = prep["feature_names"]
    missing = [feature for feature in features if feature not in normalized.columns]
    if missing:
        raise ValueError(
            "الأعمدة التالية غير موجودة في الملف: " + ", ".join(missing)
            + ". ارفعي ملف CSV يحتوي خصائص التدفق العشر المطلوبة، مثل ملف CIC-IDS بعد استخراج الخصائص."
        )

    values = normalized.loc[:, features].copy()
    for col in features:
        values[col] = pd.to_numeric(values[col], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)

    # Exclude rows with no usable input features rather than predicting from 10 imputed values.
    valid_mask = values.notna().any(axis=1).to_numpy()
    valid_positions = np.flatnonzero(valid_mask)
    skipped_rows = int((~valid_mask).sum())
    if len(valid_positions) == 0:
        return pd.DataFrame(), skipped_rows, len(missing)

    X = values.iloc[valid_positions].to_numpy(dtype=float)
    medians = np.asarray(prep["imputer_statistics"], dtype=float)
    means = np.asarray(prep["scaler_mean"], dtype=float)
    scales = np.asarray(prep["scaler_scale"], dtype=float)
    scales = np.where(scales == 0, 1.0, scales)

    # Equivalent to the saved median SimpleImputer followed by the saved StandardScaler.
    X = np.where(np.isnan(X), medians, X)
    X = (X - means) / scales

    probabilities = model.predict_proba(X)
    pred_indices = np.argmax(probabilities, axis=1).astype(int)
    labels = np.asarray(prep["label_classes"], dtype=object)[pred_indices]
    confidence = probabilities[np.arange(len(pred_indices)), pred_indices] * 100.0

    result = raw_df.iloc[valid_positions].copy().reset_index(drop=True)
    result["Predicted_Class"] = labels
    result["Model_Confidence_Percent"] = np.round(confidence, 2)
    result["Alert"] = np.where(
        labels == "BENIGN",
        "No alert (model prediction)",
        np.where(labels == "Bot", "ALERT: possible Botnet", "ALERT: possible Slowloris"),
    )
    return result, skipped_rows, len(missing)


def show_results(results: pd.DataFrame, skipped_rows: int = 0, key_prefix: str = "csv"):
    if results is None or results.empty:
        st.warning("لم توجد صفوف تحتوي على خصائص رقمية صالحة للتحليل.")
        return

    pred = results["Predicted_Class"].astype(str)
    is_alert = pred.ne("BENIGN")
    total = len(results)
    alerts = int(is_alert.sum())
    benign = int((~is_alert).sum())
    bot = int((pred == "Bot").sum())
    slow = int((pred == "DoS_Slowloris").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("التدفقات التي حُللت", f"{total:,}")
    c2.metric("تنبيهات صادرة عن النموذج", f"{alerts:,}")
    c3.metric("Botnet محتمل", f"{bot:,}")
    c4.metric("Slowloris محتمل", f"{slow:,}")

    if alerts:
        st.error(f"أصدر النموذج {alerts:,} تنبيهًا محتملًا. هذه تنبؤات آلية وليست إثباتًا لوقوع هجوم.")
    else:
        st.success("لم يصنّف النموذج أي تدفق على أنه هجوم في هذه الدفعة؛ هذا لا يثبت خلو الشبكة من الهجمات.")
    if skipped_rows:
        st.warning(f"تجاهلنا {skipped_rows:,} صفًا لأن جميع خصائص النموذج العشر كانت مفقودة أو غير رقمية.")

    st.subheader("ملخص التصنيفات")
    class_order = ["BENIGN", "Bot", "DoS_Slowloris"]
    counts = pred.value_counts().reindex(class_order, fill_value=0)
    st.bar_chart(counts.rename(index={
        "BENIGN": "طبيعي حسب النموذج",
        "Bot": "Botnet محتمل",
        "DoS_Slowloris": "Slowloris محتمل",
    }))

    only_alerts = st.checkbox("عرض التنبيهات المحتملة فقط", value=True, key=f"only_alerts_{key_prefix}")
    view = results.loc[is_alert].copy() if only_alerts else results.copy()
    display_names = {
        "Predicted_Class": "التصنيف المتوقع",
        "Model_Confidence_Percent": "ثقة النموذج (%) — غير معايرة",
        "Alert": "الحالة",
    }
    view = view.rename(columns=display_names)
    st.subheader("نتائج التدفقات")
    st.dataframe(view.head(500), use_container_width=True, hide_index=True)
    if len(view) > 500:
        st.caption("عُرضت أول 500 نتيجة فقط في الجدول لتقليل الحمل؛ ملف التنزيل يحتوي جميع النتائج المعروضة حسب التحليل.")

    csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "تنزيل جميع نتائج التحليل CSV",
        data=csv_bytes,
        file_name="ids_predictions.csv",
        mime="text/csv",
        key=f"download_results_{key_prefix}",
    )


try:
    model, prep = load_artifacts()
except Exception as exc:
    st.error("تعذر تحميل النموذج أو ملف المعالجة المسبقة.")
    st.exception(exc)
    st.stop()

st.title("🛡️ نظام كشف التسلل للشبكات وإنترنت الأشياء")
st.subheader("Hybrid ML-based IDS | Botnet & Slowloris")
st.write("لوحة تجريبية لتحليل خصائص تدفقات الشبكة باستخدام نموذج XGBoost الخفيف ذي الخصائص العشر المختارة.")

st.warning(
    "مهم أكاديميًا: نتائج الاختبار المرفقة مع النموذج ما زالت ضعيفة للكشف عن الهجمات "
    "(Accuracy للنموذج الخفيف ≈ 67.8%، وMacro-F1 ≈ 0.270، وBalanced Accuracy ≈ 0.466). "
    "كما أن مجموعة الاختبار الحالية لا تحتوي أمثلة Slowloris؛ لذلك لا يجوز اعتبار هذه اللوحة نظامًا أمنيًا جاهزًا للإنتاج. "
    "يجب إعادة تقييم البيانات وإصلاح تغطية الفئات قبل تقديم ادعاء موثوق عن الكشف."
)
st.info(
    "مصدر البيانات المطلوب: ملف CSV يحتوي خصائص تدفق الشبكة المحسوبة مسبقًا. "
    "هذه الواجهة لا تلتقط الحزم من جهاز IoT ولا تراقب الشبكة مباشرة."
)

csv_tab, manual_tab = st.tabs(["تحليل ملف تدفقات CSV", "تحليل تدفق واحد يدويًا"])

with csv_tab:
    st.write("ارفعي CSV من CIC-IDS أو ملفًا آخر، بشرط احتوائه على جميع الخصائص العشر التالية بعد توحيد أسماء الأعمدة:")
    st.code(", ".join(prep["feature_names"]), language="text")
    uploaded = st.file_uploader("اختاري ملف CSV", type=["csv"], key="csv_input")
    max_rows = st.slider("أقصى عدد صفوف للتحليل", min_value=100, max_value=100000, value=10000, step=100, key="row_limit")

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("last_file_hash") != file_hash:
            st.session_state.pop("csv_results", None)
            st.session_state["last_file_hash"] = file_hash
        st.caption(f"الملف: {uploaded.name} — الحجم: {len(file_bytes) / (1024 * 1024):.2f} MB")
        if st.button("تحليل الملف وإظهار التنبيهات", type="primary", key="analyze_csv"):
            try:
                import io
                data = pd.read_csv(io.BytesIO(file_bytes), nrows=max_rows + 1, low_memory=False)
                truncated = len(data) > max_rows
                if truncated:
                    data = data.iloc[:max_rows].copy()
                results, skipped, _ = predict_flows(data, model, prep)
                st.session_state["csv_results"] = results
                st.session_state["csv_skipped"] = skipped
                st.session_state["csv_truncated"] = truncated
                st.session_state["csv_source_name"] = uploaded.name
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("csv_results") is not None:
        if st.session_state.get("csv_truncated"):
            st.warning(f"تم تحليل أول {max_rows:,} صف فقط. قلّلي/ارفعي الحد ثم أعيدي التحليل إذا لزم.")
        show_results(st.session_state["csv_results"], st.session_state.get("csv_skipped", 0), key_prefix="csv")

with manual_tab:
    st.write("أدخلي قيمًا حقيقية محسوبة لتدفق واحد. لا تتركي أي حقل فارغًا؛ القيم المصطنعة أو الأصفار الافتراضية قد تعطي نتيجة مضللة.")
    translations = {
        "flow_duration": "مدة التدفق (flow_duration)",
        "flow_iat_mean": "متوسط الفاصل الزمني للتدفق (flow_iat_mean)",
        "flow_iat_min": "أقل فاصل زمني للتدفق (flow_iat_min)",
        "fwd_iat_max": "أكبر فاصل زمني أمامي (fwd_iat_max)",
        "bwd_iat_max": "أكبر فاصل زمني عكسي (bwd_iat_max)",
        "bwd_iat_min": "أقل فاصل زمني عكسي (bwd_iat_min)",
        "flow_iat_max": "أكبر فاصل زمني للتدفق (flow_iat_max)",
        "bwd_iat_std": "انحراف الفواصل الزمنية العكسية (bwd_iat_std)",
        "fwd_iat_min": "أقل فاصل زمني أمامي (fwd_iat_min)",
        "down/up_ratio": "نسبة الحزم/البيانات الهابطة إلى الصاعدة (down/up_ratio)",
    }
    with st.form("manual_flow_form"):
        inputs = {}
        columns = st.columns(2)
        for i, feature in enumerate(prep["feature_names"]):
            with columns[i % 2]:
                inputs[feature] = st.text_input(
                    translations.get(feature, feature),
                    value="",
                    placeholder="أدخلي قيمة رقمية فعلية",
                    key=f"manual_{feature}",
                )
        submitted = st.form_submit_button("تصنيف هذا التدفق", type="primary")

    if submitted:
        errors = []
        row = {}
        for feature, value in inputs.items():
            try:
                if not value.strip():
                    raise ValueError
                number = float(value.strip())
                if not math.isfinite(number):
                    raise ValueError
                row[feature] = number
            except ValueError:
                errors.append(feature)
        st.session_state.pop("manual_results", None)
        if errors:
            st.error("أدخلي رقمًا محدودًا وصالحًا لكل الحقول التالية: " + ", ".join(errors))
        else:
            try:
                one = pd.DataFrame([row])
                result, skipped, _ = predict_flows(one, model, prep)
                st.session_state["manual_results"] = result
                st.session_state["manual_skipped"] = skipped
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("manual_results") is not None:
        show_results(st.session_state["manual_results"], st.session_state.get("manual_skipped", 0), key_prefix="manual")

with st.expander("ملاحظات وحدود الاستخدام"):
    st.markdown(
        """
        - هذه الواجهة تصنّف **ملخصات تدفق الشبكة**؛ لا تقرأ PCAP ولا تلتقط الحزم من الشبكة مباشرة.
        - قيمة الثقة هي أعلى قيمة `predict_proba` للنموذج، وليست احتمالًا أمنيًا معايرًا.
        - الملف الناتج يحتوي التصنيف والتنبؤات، ويُنصح بإرفاقه بتقرير التقييم الأكاديمي.
        - للاستخدام الحقيقي يلزم جمع التدفقات من بوابة مراقبة داخل الشبكة، واختبار النموذج على مجموعة مستقلة تحتوي أمثلة كافية من كل فئة.
        - لا ترفعي سجلات شبكة حساسة أو عناوين IP أو بيانات خاصة إلى مستودع عام.
        """
    )

st.caption("Prototype for academic demonstration — not a production security control.")
