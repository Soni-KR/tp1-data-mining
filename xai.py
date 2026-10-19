"""SHAP and LIME explanations for fitted EduGuard pipelines (class 0)."""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


def _owners(model):
    """Map encoded columns back to input features, including one-hot categories."""
    columns = list(model.feature_names_in_)
    owners = []
    for name in model.named_steps['preprocessing'].get_feature_names_out():
        name = name.split('__', 1)[-1]
        matches = [c for c in columns if name == c or name.startswith(c + '_')]
        if not matches:
            raise ValueError(f'Cannot map encoded feature {name}')
        owners.append(columns.index(max(matches, key=len)))
    return columns, np.asarray(owners)


def _aggregate(values, owners, size):
    result = np.zeros((len(values), size))
    for j, owner in enumerate(owners):
        result[:, owner] += values[:, j]
    return result


def _space(model, background, row=None):
    """LIME needs dense numbers and zero-based category codes, not UCI codes."""
    numeric = []
    fills = {}
    for name, transformer, columns in model.named_steps['preprocessing'].transformers_:
        if name == 'num':
            numeric = list(columns)
            if hasattr(transformer, 'named_steps') and 'impute' in transformer.named_steps:
                fills = dict(zip(numeric, transformer.named_steps['impute'].statistics_))
    categories = {}
    for i, col in enumerate(background):
        if col not in numeric:
            series = background[col] if row is None else pd.concat([background[col], row[col]])
            categories[i] = list(pd.unique(series))

    def encode(frame):
        out = np.empty(frame.shape, dtype=float)
        for i, col in enumerate(background):
            if i in categories:
                values = categories[i]
                out[:, i] = [next(j for j, v in enumerate(values)
                                     if (pd.isna(v) and pd.isna(x)) or v == x)
                             for x in frame[col]]
            else:
                fill = fills.get(col, background[col].median())
                out[:, i] = frame[col].fillna(0 if pd.isna(fill) else fill)
        return out

    def decode(values):
        frame = pd.DataFrame(values, columns=background.columns)
        for i, levels in categories.items():
            frame[background.columns[i]] = [levels[int(round(v))] for v in values[:, i]]
        return frame

    return encode, decode, categories


def _permutation_shap(model, background, rows):
    """Monte Carlo interventional Shapley values using paired feature orders.

    Each path replaces background features with the individual's features.
    Averaging marginal changes over orders estimates the Shapley formula.
    Complete paths preserve additivity even with a small sampling budget.
    """
    reference = background.sample(min(8, len(background)), random_state=42)
    class_index = list(model.classes_).index(0)
    base = float(model.predict_proba(reference)[:, class_index].mean())
    rng = np.random.default_rng(42)
    values = np.zeros(rows.shape)
    for i in range(len(rows)):
        orders, states = [], []
        for _ in range(4):
            order = rng.permutation(len(rows.columns))
            for sequence in (order, order[::-1]):
                orders.append(sequence)
                frame = reference.copy()
                states.append(frame.copy())
                for index in sequence:
                    frame[rows.columns[index]] = rows.iloc[i, index]
                    states.append(frame.copy())
        probabilities = model.predict_proba(pd.concat(states, ignore_index=True))[:, class_index]
        outputs = probabilities.reshape(len(orders), len(rows.columns) + 1, len(reference)).mean(axis=2)
        for sequence, differences in zip(orders, np.diff(outputs, axis=1)):
            values[i, sequence] += differences / len(orders)
    return values, np.full(len(rows), base)


def shap_values(model, background, rows):
    """Return input-level attributions, always oriented toward class 0."""
    columns, owners = _owners(model)
    pre, estimator = model.named_steps['preprocessing'], model.named_steps['model']
    transformed = pre.transform(rows)
    class_index = list(model.classes_).index(0)
    sign = -1 if class_index == 0 else 1
    if hasattr(estimator, 'get_booster'):
        # Exact TreeSHAP is built into XGBoost; no second model is fitted.
        from xgboost import DMatrix
        matrix = DMatrix(transformed)
        contributions = estimator.get_booster().predict(matrix, pred_contribs=True)
        values, base = sign * contributions[:, :-1], sign * contributions[:, -1]
        units, method = 'log-odds', 'TreeSHAP (XGBoost)'
    elif hasattr(estimator, 'coef_'):
        # Exact independent-feature linear SHAP in the encoded input space.
        mean = np.asarray(pre.transform(background)).mean(axis=0)
        coefficients = sign * estimator.coef_[0]
        values = (np.asarray(transformed) - mean) * coefficients
        base = np.full(len(rows), sign * estimator.intercept_[0] + mean @ coefficients)
        units, method = 'log-odds', 'Linear SHAP (independent encoded features)'
    else:
        values, base = _permutation_shap(model, background, rows)
        owners = np.arange(len(columns))
        units, method = 'probability', 'Permutation SHAP (8 sampled orders, up to 8 training background rows)'
    grouped = _aggregate(np.asarray(values), owners, len(columns))
    base = np.broadcast_to(np.asarray(base), (len(rows),))
    output = base + grouped.sum(axis=1)
    probability = expit(output) if units == 'log-odds' else output
    expected = model.predict_proba(rows)[:, class_index]
    if not np.allclose(probability, expected, atol=2e-5):
        raise ValueError('SHAP contributions do not reconstruct class-0 probabilities')
    return {'values': grouped.tolist(), 'base_values': base.tolist(),
            'units': units, 'method': method, 'features': columns}


def global_explanation(model, background, rows, labels=None):
    """Same ROC-AUC permutation metric and seed for every classifier."""
    columns, owners = _owners(model)
    estimator = model.named_steps['model']
    native = None
    native_method = 'Not available for this model (use permutation importance)'
    if hasattr(estimator, 'coef_'):
        raw = np.abs(estimator.coef_[0])
        native_method = 'Absolute coefficients (encoded/scaled inputs; grouped by source feature)'
    elif hasattr(estimator, 'feature_importances_'):
        raw = estimator.feature_importances_
        native_method = 'Gain' if hasattr(estimator, 'get_booster') else 'Impurity decrease'
    else:
        raw = None
    if raw is not None:
        native = _aggregate(np.asarray(raw)[None, :], owners, len(columns))[0].tolist()
    permutation = None
    if labels is not None:
        def score(estimator, data, y):
            return roc_auc_score(np.asarray(y) == 0, estimator.predict_proba(data)[:, list(estimator.classes_).index(0)])
        result = permutation_importance(model, rows, labels, scoring=score, n_repeats=3, random_state=42, n_jobs=1)
        permutation = {'mean': result.importances_mean.tolist(), 'std': result.importances_std.tolist(),
                       'metric': 'Decrease in ROC-AUC; 3 shuffles per input feature'}
    shap = shap_values(model, background, rows)
    return {'features': columns, 'native': native, 'native_method': native_method,
            'permutation': permutation, 'shap': shap,
            'data': rows.astype(object).where(rows.notna(), None).values.tolist(),
            'sample_size': len(rows)}


def local_explanation(model, background, row, class_label='class 0'):
    from lime.lime_tabular import LimeTabularExplainer
    columns = list(model.feature_names_in_)
    row = row[columns]
    shap = shap_values(model, background, row)
    encode, decode, categories = _space(model, background, row)
    class_index = list(model.classes_).index(0)
    explainer = LimeTabularExplainer(
        encode(background), feature_names=columns, mode='classification',
        class_names=[str(c) for c in model.classes_], categorical_features=list(categories),
        categorical_names={i: [str(v) for v in levels] for i, levels in categories.items()},
        discretize_continuous=True, random_state=42,
    )
    explanation = explainer.explain_instance(
        encode(row)[0], lambda data: model.predict_proba(decode(data)),
        labels=(class_index,), num_features=len(columns), num_samples=3000,
    )
    weights = np.zeros(len(columns))
    for index, weight in explanation.local_exp[class_index]:
        weights[index] = weight
    probability = float(model.predict_proba(row)[0, class_index])
    contributions = np.asarray(shap['values'][0])
    strongest = np.argsort(-np.abs(contributions))[:3]
    names = {'Curricular units 1st sem (approved)': 'first-semester units passed',
             'Curricular units 1st sem (enrolled)': 'first-semester units enrolled',
             'Curricular units 1st sem (grade)': 'first-semester average grade',
             'Tuition fees up to date': 'tuition fees paid on time'}
    factors = [f'{names.get(columns[i], columns[i])}={row.iloc[0, i]} ({"raises" if contributions[i] > 0 else "lowers"} the estimate)'
               for i in strongest if abs(contributions[i]) > 1e-8]
    top_count = min(5, len(columns))
    top_shap = set(np.argsort(-np.abs(contributions))[:top_count])
    top_lime = set(np.argsort(-np.abs(weights))[:top_count])
    shared = sorted(top_shap & top_lime)
    comparison = [{'feature': columns[i], 'same_direction': bool(np.sign(contributions[i]) == np.sign(weights[i]))}
                  for i in shared]
    return {'shap': shap, 'lime': {'weights': weights.tolist(),
            'rules': explanation.as_list(label=class_index), 'intercept': float(explanation.intercept[class_index]),
            'local_prediction': float(explanation.local_pred[0]), 'r2': float(explanation.score),
            'absolute_error': abs(float(explanation.local_pred[0]) - probability), 'samples': 3000},
            'comparison': comparison, 'top_count': top_count,
            'summary': f'Estimated {class_label} probability: {probability:.1%}. Main model contributions: '
                       + ('; '.join(factors) if factors else 'no material feature contribution') + '.',
            'note': 'SHAP and LIME use different baselines and units: compare directions and rankings, not magnitudes. '
                    'LIME is a local approximation; low R² or a large prediction error reduces confidence. '
                    'Its perturbations may combine implausible input values. '
                    'Contributions describe associations in this model, not causal effects.'}
