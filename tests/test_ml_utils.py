import pandas as pd

def test_custom_binary_dataset():
    from ml_utils import train_models, metrics
    df = pd.DataFrame({'score': list(range(60)), 'category': ['a', 'b']*30, 'outcome': ['risk']*30+['safe']*30})
    df.loc[3, 'score'] = None
    config = {'features': ['score','category'], 'numeric':['score'], 'target':'outcome', 'positive':'risk', 'test_size':0.2, 'scaling':'Standard', 'imputation':'median'}
    result = train_models(df, config, ['Logistic Regression'], False, 10, 3, 5, 1.0)
    assert len(result['y_test']) == 12
    assert len(result['models']['Logistic Regression']['oof']) == 48
    measured = metrics(result['y_test'], result['models']['Logistic Regression']['test'], .5)
    assert 0 <= measured['ROC-AUC'] <= 1
