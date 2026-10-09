"""Run with streamlit run app.py."""
from pathlib import Path
from io import BytesIO
import os
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

PATH = Path(__file__).resolve().parent
NUMERIC = ['Application order', 'Previous qualification (grade)', 'Admission grade', 'Age at enrollment', 'Unemployment rate', 'Inflation rate', 'GDP']
MODELS = ['Logistic Regression', 'Decision Tree', 'KNN', 'Random Forest', 'XGBoost']
st.set_page_config(page_title='EduGuard', page_icon='🎓', layout='wide')
st.markdown('<style>.stApp{background:#f6f8fc}h1,h2,h3{color:#102b50}div[data-testid="stMetric"]{background:white;padding:16px;border-radius:10px}</style>', unsafe_allow_html=True)
st.title('🎓 EduGuard')
st.caption('First-semester student dropout prediction & decision support · ENSI')
page = st.sidebar.radio('Navigate', ['Overview', 'Data exploration', 'Preprocessing', 'Model comparison', 'Evaluation', 'Student prediction', 'About'])
default_url = os.getenv('EDUGUARD_API_URL', 'http://127.0.0.1:8000')
try:
    default_url = st.secrets.get('EDUGUARD_API_URL', default_url)
except st.errors.StreamlitSecretNotFoundError:
    pass
api_url = st.sidebar.text_input('FastAPI URL', default_url)
uploaded_csv = st.sidebar.file_uploader('Optional labeled CSV', type='csv')
if st.sidebar.button('Check API readiness'):
    try:
        response = requests.get(api_url.rstrip('/') + '/health', timeout=30)
        if response.ok:
            st.sidebar.success('Model is ready')
        else:
            st.sidebar.error('API is running, but the model is unavailable.')
            st.sidebar.json(response.json())
    except requests.RequestException:
        st.sidebar.error('Cannot reach FastAPI. Check its URL and start the backend.')

@st.cache_data
def reference():
    return pd.read_csv(PATH / 'reference.csv')

def show_table(frame, **kwargs):
    st.dataframe(frame, **kwargs)

@st.cache_data(ttl=3600)
def public_data():
    local = PATH / 'data.csv'
    if local.exists():
        return pd.read_csv(local, sep=None, engine='python')
    url = 'https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.endswith('.csv'))
        return pd.read_csv(archive.open(name), sep=';')

def dataset():
    try:
        df = pd.read_csv(uploaded_csv, sep=None, engine='python') if uploaded_csv else public_data()
    except Exception as exc:
        st.error(f'Could not load the labeled dataset. Upload a CSV or add data.csv. {exc}')
        st.stop()
    df.columns = df.columns.str.strip()
    target = st.sidebar.selectbox('Target column', list(df.columns), index=list(df.columns).index('Target') if 'Target' in df else len(df.columns)-1)
    df = df.dropna(subset=[target]).copy()
    if target == 'Target' and set(df[target].unique()) <= {'Dropout', 'Enrolled', 'Graduate'}:
        df = df[df[target] != 'Enrolled'].copy()
        df[target] = df[target].map({'Dropout': 0, 'Graduate': 1})
        st.sidebar.caption('Enrolled excluded. Dropout=0; Graduate=1.')
    df = df.drop(columns=[c for c in df if 'Curricular units 2nd sem' in c])
    classes = list(df[target].unique())
    if len(classes) != 2:
        st.error('Choose a binary target with exactly two observed classes.')
        st.stop()
    original = target == 'Target' and set(reference().columns) <= set(df.columns) and set(classes) == {0,1}
    positive = st.sidebar.selectbox('Class of interest (encoded as 0)', classes, index=classes.index(0) if 0 in classes else 0, disabled=original)
    return df, target, positive

if page == 'Overview':
    st.subheader('Identify students who may benefit from support')
    st.write('Estimate dropout risk at the end of semester one, allowing academic follow-up before semester two. Predictions are estimates, not certain outcomes.')
    for col, label, value in zip(st.columns(4), ['Students retained', 'Original input features', 'Dropout', 'Graduate'], ['3,630', '30', '1,421 · 39.1%', '2,209 · 60.9%']):
        col.metric(label, value)
    st.subheader('Historical notebook results')
    st.caption('These are existing notebook measurements, not results from this dashboard run.')
    show_table(pd.DataFrame({'Model': MODELS, 'CV ROC-AUC': [0.9348,0.8924,0.8775,0.9303,0.9351], 'Test ROC-AUC': [0.9406,0.9038,0.8762,0.9394,0.9404], 'Dropout F1': [0.8587,0.8302,0.6591,0.8387,0.8717]}), hide_index=True, width='stretch')
    st.info('Saved model: XGBoost. Historical dropout precision 0.9390; recall 0.8134.')

elif page in ['Data exploration', 'Preprocessing', 'Model comparison']:
    df, target, positive = dataset()
    # Invalidate experiments when data or target settings change.
    signature = (int(pd.util.hash_pandas_object(df, index=True).sum()), target, str(positive))
    if st.session_state.get('dataset_signature') != signature:
        st.session_state.pop('results', None)
        st.session_state.pop('config', None)
        st.session_state.dataset_signature = signature
    if page == 'Data exploration':
        st.subheader('Explore the labeled data')
        show_table(df.head(100), width='stretch')
        st.write(f'{len(df):,} rows · {len(df.columns)-1} features · {df.duplicated().sum()} duplicate rows')
        show_table(df.describe(include='all').T, width='stretch')
        st.subheader('Missing values')
        show_table(df.isna().sum().rename('Missing').to_frame())
        st.plotly_chart(px.histogram(df, x=target, title='Target distribution'), width='stretch')
        feature = st.selectbox('Explore feature', [c for c in df if c != target])
        st.plotly_chart(px.histogram(df, x=feature, color=df[target].astype(str), barmode='overlay'), width='stretch')
        corr = df.select_dtypes('number').corr()
        st.plotly_chart(px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title='Numeric correlations'), width='stretch')
        grouped = df.assign(interest=(df[target] == positive).astype(int)).groupby(feature, observed=True)['interest'].agg(['mean','count']).reset_index()
        st.plotly_chart(px.bar(grouped, x=feature, y='mean', hover_data=['count'], title=f'Observed rate of class {positive} (association, not causation)'), width='stretch')
    else:
        candidates = [c for c in df if c != target]
        default_numeric = [c for c in candidates if c in NUMERIC or 'Curricular units 1st sem' in c] if set(reference().columns) <= set(candidates) else list(df[candidates].select_dtypes('number').columns)
        saved = st.session_state.get('config', {})
        with st.expander('Preprocessing configuration', expanded=page == 'Preprocessing'):
            selected = st.multiselect('Features', candidates, default=saved.get('features', candidates))
            numeric = st.multiselect('Numerical features (others are categorical / binary)', selected, default=[c for c in saved.get('numeric', default_numeric) if c in selected])
            scaling_options = ['Standard', 'MinMax', 'None']
            scaling = st.selectbox('Scaling', scaling_options, index=scaling_options.index(saved.get('scaling', 'Standard')))
            imputation_options = ['median', 'mean', 'most_frequent']
            imputation = st.selectbox('Numerical missing-value treatment', imputation_options, index=imputation_options.index(saved.get('imputation', 'median')))
            st.caption('Categorical values: most-frequent imputation + one-hot encoding with unknown categories ignored. Original binary indicators are preserved.')
            test_size = st.slider('Held-out test percentage', 10, 40, int(saved.get('test_size', 0.2)*100))/100
        config = dict(features=selected, numeric=numeric, scaling=scaling, imputation=imputation, test_size=test_size, target=target, positive=positive)
        st.info('Stratified split with random_state=42. Imputation, scaling and encoding are fitted inside each cross-validation pipeline. No second-semester features are available.')
        if page == 'Preprocessing' and st.button('Save preprocessing settings', disabled=not selected):
            st.session_state.config = config
            st.success('Settings saved for Model comparison.')
        if page == 'Model comparison':
            names = st.multiselect('Models', MODELS, default=['Logistic Regression', 'XGBoost'])
            a,b,c,d = st.columns(4)
            trees = a.number_input('Trees', 10, 500, 100)
            depth = b.number_input('Maximum depth', 1, 20, 5)
            neighbors = c.number_input('KNN neighbors', 1, 30, 5)
            regularization = d.number_input('Logistic C', 0.01, 100.0, 1.0)
            tune = st.checkbox('Run five-fold GridSearchCV (slower)')
            if st.button('Train and compare', type='primary', disabled=not names or not selected):
                try:
                    from ml_utils import train_models
                    with st.spinner('Training pipelines and computing training out-of-fold probabilities…'):
                        st.session_state.results = train_models(df, config, names, tune, trees, depth, neighbors, regularization)
                    st.session_state.config = config
                except Exception as exc:
                    st.error(f'Training failed: {exc}. Check your CSV and preprocessing settings.')
            if 'results' in st.session_state:
                show_table(pd.DataFrame([{'Model': name, 'CV ROC-AUC': result['cv_auc']} for name,result in st.session_state.results['models'].items()]), hide_index=True)
                for name,result in st.session_state.results['models'].items():
                    with st.expander(f'{name} parameters'):
                        st.json(result['parameters'])
                st.caption('Results persist until the data changes or Train is clicked again. They reflect the last training configuration.')

elif page == 'Evaluation':
    st.subheader('Threshold exploration and final evaluation')
    if 'results' not in st.session_state:
        st.info('Train models on the Model comparison page first. reference.csv has no labels and cannot be used to measure accuracy.')
    else:
        try:
            from ml_utils import metrics
            from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
            results = st.session_state.results
            st.write(f'Class of interest: {results["positive"]} (encoded as 0)')
            threshold = st.slider('Probability threshold', 0.0, 1.0, 0.5, 0.01)
            final = st.checkbox('Show held-out test results at this fixed threshold')
            st.warning('Choose thresholds using training out-of-fold results. Once test results are viewed, do not keep tuning against them. With GridSearchCV, these training estimates are exploratory because hyperparameters were selected on the same training folds.')
            key = 'test' if final else 'oof'
            y = results['y_test' if final else 'y_train']
            st.caption('Held-out test data' if final else 'Out-of-fold training data')
            show_table(pd.DataFrame({name: metrics(y, r[key], threshold) for name,r in results['models'].items()}).T)
            roc, pr = go.Figure(), go.Figure()
            for name,r in results['models'].items():
                fpr,tpr,_ = roc_curve(y == 0, r[key])
                precision,recall,_ = precision_recall_curve(y == 0, r[key])
                roc.add_scatter(x=fpr,y=tpr,name=name)
                pr.add_scatter(x=recall,y=precision,name=name)
            roc.update_layout(title='ROC curves', xaxis_title='False positive rate', yaxis_title='Recall')
            pr.update_layout(title='Precision–recall curves', xaxis_title='Recall', yaxis_title='Precision')
            left,right = st.columns(2)
            left.plotly_chart(roc, width='stretch')
            right.plotly_chart(pr, width='stretch')
            name = st.selectbox('Confusion matrix model', list(results['models']))
            matrix = confusion_matrix(y, np.where(results['models'][name][key] >= threshold, 0, 1), labels=[0,1])
            st.plotly_chart(px.imshow(matrix, text_auto=True, x=['Class of interest','Other class'], y=['Class of interest','Other class'], labels={'x':'Predicted','y':'Actual'}, color_continuous_scale='Blues'), width='stretch')
        except Exception as exc:
            st.error(f'Evaluation unavailable: {exc}')

elif page == 'Student prediction':
    st.subheader('Estimate student dropout risk')
    df = reference()
    mode = st.radio('Student profile', ['Existing reference profile', 'Fictional / edited profile'], horizontal=True)
    index = st.number_input('Reference row (not a student identity)', 0, len(df)-1, 0)
    data = df.iloc[index].to_dict()
    st.caption('Reference rows have no ground-truth labels. Input limits and categorical choices come from observed reference values.')
    with st.form('student'):
        if mode == 'Fictional / edited profile':
            columns = st.columns(3)
            for i,feature in enumerate(df.columns):
                values = sorted(df[feature].dropna().unique().tolist())
                with columns[i % 3]:
                    if set(values) <= {0,1}:
                        options = {0:'No (0)',1:'Yes (1)'}
                        if feature == 'Gender': options = {0:'Female (0)',1:'Male (1)'}
                        if feature == 'Daytime/evening attendance': options = {0:'Evening (0)',1:'Daytime (1)'}
                        data[feature] = st.selectbox(feature, values, index=values.index(data[feature]), format_func=lambda v: options[v])
                    elif feature in NUMERIC or 'Curricular units 1st sem' in feature:
                        data[feature] = st.number_input(feature, min_value=float(values[0]), max_value=float(values[-1]), value=float(data[feature]))
                    else:
                        labels = {1:'Single',2:'Married',3:'Widower',4:'Divorced',5:'Facto union',6:'Legally separated'} if feature == 'Marital status' else {}
                        data[feature] = st.selectbox(feature, values, index=values.index(data[feature]), format_func=lambda v: f'{labels[v]} ({int(v)})' if v in labels else f'Dataset code {int(v)}')
        else:
            show_table(pd.DataFrame([data]), width='stretch')
        threshold = st.slider('Dropout decision threshold', 0.0, 1.0, 0.5, 0.01)
        submitted = st.form_submit_button('Predict', type='primary')
    if submitted:
        try:
            response = requests.post(
                api_url.rstrip('/') + '/predict',
                json={'data': {k:float(v) for k,v in data.items()}, 'threshold': threshold},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            a,b,c = st.columns(3)
            a.metric('Estimated dropout probability', f'{result["dropout_probability"]:.1%}')
            b.metric('Estimated graduation probability', f'{result["graduation_probability"]:.1%}')
            c.metric('Risk category', result['risk_level'])
            st.write(f'Predicted outcome: **{result["prediction"]}**, using threshold {result["threshold"]:.2f}.')
            st.info('This estimate is not a confirmed future outcome. Students with increased risk may benefit from academic counseling, tutoring, financial-support assessment or supportive outreach.')
        except requests.RequestException as exc:
            st.error('Prediction unavailable. Check API readiness and the FastAPI URL in the sidebar.')
            if exc.response is not None:
                st.code(exc.response.text)
    st.caption('Application categories: Low <30%; Medium 30–<70%; High ≥70%. These boundaries are not statistically validated and do not change with the decision threshold.')

else:
    st.write('Dataset: Predict Students’ Dropout and Academic Success (UCI / Kaggle). The original dataset has 4,424 students and 36 input features. Excluding Enrolled leaves 3,630 students; excluding semester-two variables leaves 30 inputs.')
    st.write('The existing notebook compares Logistic Regression, Decision Tree, KNN, Random Forest and XGBoost. The saved XGBoost pipeline includes preprocessing. The prediction page calls FastAPI through HTTP and never loads the model itself.')
    st.write('Use as decision support, never as an automatic basis for excluding or penalizing students. Historical cohort results may not generalize to other institutions; probabilities are not established as calibrated. Audit fairness and obtain appropriate authorization before using real student records.')
