"""Optional dashboard experiments; never overwrite the saved prediction model."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

NUMERICAL = ['Application order', 'Previous qualification (grade)', 'Admission grade', 'Age at enrollment', 'Unemployment rate', 'Inflation rate', 'GDP']
BINARY = ['Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest', 'XGBoost']

def train_models(df, config, names, tune, trees, depth, neighbors, regularization):
    X = df[config['features']].copy()
    y = (df[config['target']] != config['positive']).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], stratify=y, random_state=42)
    numeric = config['numeric']
    binary = [c for c in X if c not in numeric and c in BINARY and set(X_train[c].dropna().unique()) <= {0, 1}]
    categorical = [c for c in X if c not in numeric + binary]
    steps = [('impute', SimpleImputer(strategy=config['imputation']))]
    if config['scaling'] != 'None':
        steps.append(('scale', StandardScaler() if config['scaling'] == 'Standard' else MinMaxScaler()))
    preprocessor = ColumnTransformer([
        ('num', Pipeline(steps), numeric),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
        ('binary', SimpleImputer(strategy='most_frequent'), binary),
    ])
    models = {
        'Logistic Regression': LogisticRegression(C=regularization, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=depth, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=neighbors),
        'Random Forest': RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=42, n_jobs=1),
        'XGBoost': XGBClassifier(n_estimators=trees, max_depth=depth, eval_metric='logloss', random_state=42, n_jobs=1),
    }
    grids = {
        'Logistic Regression': {'model__C': [0.1, 1, 10]},
        'Decision Tree': {'model__max_depth': [3, 5, 10]},
        'KNN': {'model__n_neighbors': [5, 11, 21]},
        'Random Forest': {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10]},
        'XGBoost': {'model__n_estimators': [100, 200], 'model__max_depth': [3, 5], 'model__learning_rate': [0.05, 0.1]},
    }
    if y_train.value_counts().min() < 5:
        raise ValueError('Each class needs at least five training observations for five-fold CV.')
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}
    for name in names:
        pipe = Pipeline([('preprocessing', preprocessor), ('model', models[name])])
        if tune:
            search = GridSearchCV(pipe, grids[name], cv=cv, scoring='roc_auc', n_jobs=1)
            search.fit(X_train, y_train)
            pipe, score, parameters = search.best_estimator_, search.best_score_, search.best_params_
        else:
            score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc').mean()
            pipe.fit(X_train, y_train)
            parameters = models[name].get_params()
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method='predict_proba')[:, 0]
        results[name] = {'cv_auc': score, 'parameters': parameters, 'oof': oof, 'test': pipe.predict_proba(X_test)[:, 0]}
    return {'models': results, 'y_train': y_train.to_numpy(), 'y_test': y_test.to_numpy(), 'positive': str(config['positive'])}

def metrics(y, probability, threshold):
    pred = np.where(probability >= threshold, 0, 1)
    return {'ROC-AUC': roc_auc_score(y == 0, probability), 'Precision': precision_score(y, pred, pos_label=0, zero_division=0), 'Recall': recall_score(y, pred, pos_label=0, zero_division=0), 'F1': f1_score(y, pred, pos_label=0, zero_division=0), 'PR-AUC (AP)': average_precision_score(y == 0, probability), 'Flagged': int((pred == 0).sum())}
