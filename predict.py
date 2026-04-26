"""
predict.py
==========
Core prediction function + CLI interface.
No TensorFlow — uses scikit-learn model.

Run: python predict.py
"""

import numpy as np
import joblib
import os

# ── Load artefacts ─────────────────────────────────────────────
if not os.path.exists('model.joblib') or not os.path.exists('scaler.joblib'):
    raise RuntimeError("model.joblib / scaler.joblib not found. Run train_rnn.py first.")

_model  = joblib.load('model.joblib')
_scaler = joblib.load('scaler.joblib')

FEATURES  = ['attendance', 'assignment', 'quiz', 'study_hours']
TIMESTEPS = 5
N_FEAT    = len(FEATURES)


def predict_student(weekly_data: list) -> dict:
    """
    Predict Pass/Fail from 5 weeks of data.

    Parameters
    ----------
    weekly_data : list of 5 lists, each: [attendance, assignment, quiz, study_hours]
    Example:
        [[70,75,80,5],[72,78,82,6],[68,70,79,5],[75,80,85,7],[71,76,83,6]]

    Returns
    -------
    dict: result, label, pass_prob, fail_prob, interpretation
    """
    if len(weekly_data) != TIMESTEPS:
        raise ValueError(f"Need 5 weeks of data, got {len(weekly_data)}")

    flat    = np.array(weekly_data, dtype=float).flatten().reshape(1, -1)
    flat_s  = _scaler.transform(flat)

    prediction = int(_model.predict(flat_s)[0])
    probs      = _model.predict_proba(flat_s)[0]
    pass_prob  = round(float(probs[1]) * 100, 2)
    fail_prob  = round(float(probs[0]) * 100, 2)

    if prediction == 1:
        interp = ("🌟 Excellent – very high chance of passing!" if pass_prob >= 80 else
                  "✅ Good – likely to pass, keep it up!"        if pass_prob >= 65 else
                  "⚠️ Borderline pass – consistent effort needed.")
    else:
        interp = ("❌ High risk – urgent improvement required."  if fail_prob >= 80 else
                  "⚠️ Likely to fail – focus on weak areas now." if fail_prob >= 65 else
                  "🔶 At risk – small improvements can help.")

    return {"result": prediction,
            "label": "Pass" if prediction == 1 else "Fail",
            "pass_prob": pass_prob,
            "fail_prob": fail_prob,
            "interpretation": interp}


# ── CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🎓 RNN Student Evaluator (CLI)")
    print("="*55)
    weekly = []
    for w in range(1, 6):
        print(f"\n  --- Week {w} ---")
        try:
            a  = float(input("    Attendance   (0-100): "))
            b  = float(input("    Assignment   (0-100): "))
            c  = float(input("    Quiz         (0-100): "))
            d  = float(input("    Study hours  (0-15) : "))
            weekly.append([a, b, c, d])
        except ValueError:
            print("  ❌ Numbers only!"); exit(1)

    r = predict_student(weekly)
    print("\n" + "-"*55)
    print(f"  Result     : {r['label']}")
    print(f"  Pass Prob  : {r['pass_prob']}%")
    print(f"  Fail Prob  : {r['fail_prob']}%")
    print(f"  {r['interpretation']}")
    print("-"*55)
