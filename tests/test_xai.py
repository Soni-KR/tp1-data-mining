import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from ml_utils import train_models
from xai import global_explanation, local_explanation, shap_values, _permutation_shap


@pytest.fixture(scope='module')
def experiments():
    rng = np.random.default_rng(12)
    score = rng.normal(size=100)
    frame = pd.DataFrame({'score': score, 'group': np.where(np.arange(100) % 2, 'a', 'b'),
                          'outcome': np.where(score + rng.normal(0, .4, 100) < 0, 'risk', 'safe')})
    frame.loc[3, 'score'] = np.nan
    config = {'features': ['score', 'group'], 'numeric': ['score'], 'target': 'outcome',
              'positive': 'risk', 'test_size': .2, 'scaling': 'Standard', 'imputation': 'mean'}
    return train_models(frame, config, ['Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest', 'XGBoost'],
                        False, 10, 3, 5, 1.)


@pytest.mark.parametrize('name', ['Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest', 'XGBoost'])
def test_explanations_reconstruct_class_zero_and_use_same_individual(experiments, name):
    model = experiments['models'][name]['pipeline']
    rows = experiments['X_test']
    background = experiments['X_train']
    result = global_explanation(model, background, rows, experiments['y_test'])
    values = np.asarray(result['shap']['values'])
    assert values.shape == rows.shape
    output = np.asarray(result['shap']['base_values']) + values.sum(axis=1)
    reconstructed = expit(output) if result['shap']['units'] == 'log-odds' else output
    assert np.allclose(reconstructed, model.predict_proba(rows)[:, 0], atol=2e-5)
    assert len(result['permutation']['mean']) == 2
    assert result['native'] is None if name == 'KNN' else len(result['native']) == 2
    local = local_explanation(model, background, rows.iloc[[0]])
    probability = model.predict_proba(rows.iloc[[0]])[0, 0]
    assert f'{probability:.1%}' in local['summary']
    assert np.isfinite(local['lime']['weights']).all()
    assert len(local['lime']['rules']) == 2
    assert np.isfinite(local['lime']['r2'])
    assert local['lime']['absolute_error'] == pytest.approx(abs(local['lime']['local_prediction'] - probability))
    # The LIME category decoder must preserve real string categories for the pipeline.
    assert any('group' in rule for rule, _ in local['lime']['rules'])


def test_saved_tree_shap_is_dropout_not_graduation():
    import api
    rows = pd.read_csv(api.path / 'reference.csv').iloc[:10]
    result = shap_values(api.get_model(), api.background(), rows)
    output = np.asarray(result['base_values']) + np.asarray(result['values']).sum(axis=1)
    assert np.allclose(expit(output), api.get_model().predict_proba(rows)[:, 0], atol=2e-5)
    assert result['features'] == api.features


def test_sampled_shap_matches_two_feature_shapley_formula():
    from types import SimpleNamespace
    def probabilities(frame):
        p = expit(frame.a + frame.b + 2 * frame.a * frame.b).to_numpy()
        return np.column_stack([p, 1-p])
    model = SimpleNamespace(classes_=[0, 1], predict_proba=probabilities)
    background = pd.DataFrame({'a': [0., 0., 1., 1.], 'b': [0., 1., 0., 1.]})
    row = pd.DataFrame({'a': [1.], 'b': [0.]})
    values, baseline = _permutation_shap(model, background, row)
    empty = probabilities(background)[:, 0].mean()
    a = probabilities(background.assign(a=1.))[:, 0].mean()
    b = probabilities(background.assign(b=0.))[:, 0].mean()
    both = probabilities(row)[0, 0]
    expected = [.5 * (a-empty + both-b), .5 * (b-empty + both-a)]
    assert np.allclose(values[0], expected)
    assert baseline[0] == pytest.approx(empty)
