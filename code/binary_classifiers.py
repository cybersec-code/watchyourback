import pandas as pd
from joblib import dump, load
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectPercentile, RFECV
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

def search_test_idx(cids, cids_y, perc=0.2, random_state=0):
    from sklearn.utils import shuffle
    test_elems = int(sum(cids) * perc)
#    cids.sort_values(inplace=True)
    cids = shuffle(cids, random_state=random_state)
    test_v = 0
    positives = 0
    negatives = 0
    test_idx = []
    for i in cids.index:
        if test_v + cids[i] > test_elems:
            continue
        l_pos = cids[i] if cids_y[i] == 'exchange' else 0
        l_neg = cids[i] if cids_y[i] == 'non-exchange' else 0
        if positives+l_pos > test_elems/2 or negatives+l_neg > test_elems/2:
            continue
        positives += l_pos
        negatives += l_neg
        test_v += cids[i]
        test_idx.append(i)
        if test_v == test_elems:
            break
    return test_idx

def split_by_cluster_id(X, y, perc=0.2, random_state=0):
    ''' Stratified train/test split, separating elements of the same
    cluster to be either in the training or in the testing set.
    '''
    cids = X.groupby('cluster_id').apply(lambda x: len(x))
    cids_y = {c: y[X[X.cluster_id==c].index].values[0] for c in cids.keys()}
    test_idx = search_test_idx(cids, cids_y, perc, random_state)
    if not test_idx:
        print(f"Failed to split data by {perc} percent. Try a larger value.")
        raise Exception
    X_test, y_test = None, None
    X_train, y_train = None, None
    X_test = X[X.cluster_id.isin(test_idx)]
    y_test = y[X_test.index]
    X_train = X[~X.cluster_id.isin(test_idx)]
    y_train = y[X_train.index]
    return X_train, X_test, y_train, y_test

def sanitize(data):
    cluster_ids = data[data['label']=='exchange']['cluster_id']
    for c_id in cluster_ids.unique():
        print(f"Checking consistency of cluster {c_id}")
        addr_idxs = dataset[dataset['cluster_id']==c_id].index
        for idx in addr_idxs:
            if data.loc[idx]['label'] != 'exchange':
                print(f"\tChanging label of {data.loc[idx]['address']}")
                data.at[idx, 'label'] = 'exchange'

# These are not features
exclude = ['label', 'address', 'cluster_id', 'datetime_first', 'datetime_last',
       'timestamp_first', 'timestamp_last', 'timestamp_first_out',
       'timestamp_last_out', 'timestamp_first_in', 'timestamp_last_in']
#       'cluster_size']

# Read the dataset of features in TSV format
dfn = '../data/datasets/dataset.tsv'
dataset = pd.read_csv(dfn, sep='\t')
# Change the label for online-wallets (e.g. bitcoin-otc users using some
# exchange) whenever an intersection between clusters is found
sanitize(dataset)
print(f"Exchanges: {dataset[dataset.label=='exchange'].address.size}")
print(f"Non-Exchanges: {dataset[dataset.label=='non-exchange'].address.size}")

X, y = dataset.drop(['label'], axis=1), dataset['label']
labels = sorted(y.unique())

categoricals = ['type']
numerical = [n for n in X.columns if (n not in categoricals and n not in
    exclude)]

# Define transformers for categorical and numerical features
types=[list(X['type'].unique())]
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
num_transformer_pt = Pipeline(steps=[('scaler',
    PowerTransformer(method='yeo-johnson', standardize=True))])
cat_transformer = Pipeline(steps=[('onehot',
    OneHotEncoder(handle_unknown='ignore', categories=types))])

# Transform categoricals, pass through numericals
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical),
        ('cat', cat_transformer, categoricals)],
    remainder='drop', n_jobs=-1)

# Transform both numericals (StandardScaler) and categoricals
preprocessor_std = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical),
        ('cat', cat_transformer, categoricals)],
    remainder='drop', n_jobs=-1)

# Transform both numericals (PowerTransformer) and categoricals
preprocessor_pt = ColumnTransformer(
    transformers=[
        ('num', num_transformer_pt, numerical),
        ('cat', cat_transformer, categoricals)],
    remainder='drop', n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        shuffle=True, random_state=0)
#X_train, X_test, y_train, y_test = split_by_cluster_id(X, y, perc=0.2,
#        random_state=0)
print(f"Train Exchanges: {y_train[y_train=='exchange'].size}")
print(f"Train Non-Exchanges: {y_train[y_train=='non-exchange'].size}")
print(f"Test Exchanges: {y_test[y_test=='exchange'].size}")
print(f"Test Non-Exchanges: {y_test[y_test=='non-exchange'].size}")

scoring = ['precision_macro', 'recall_macro', 'f1_macro']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

final_scores = {}
feature_counts = defaultdict(int)
selected_features = {}

def print_scores(key, clf, scoring, cvscores):
    scores_dict = {}
    print(f"{clf}")
    for score in scoring:
        scores = cvscores[f"test_{score}"]
        print(f"CV Scores for {score}: {scores}")
        print("Mean: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        scores_dict.update({score: scores})
    print("----------------------------------------")
    return {key: scores_dict}

def feature_selection(X, y, preprocessor, selector):
    features = []
    w = 'without' if type(preprocessor.transformers[0][1]) == str else 'with'
    print(f"Numerical features {w} standardization")
    pipe_selector = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector)])
    pipe_selector.fit(X, y)
    support = pipe_selector.named_steps['selector'].support_
    ranking = pipe_selector.named_steps['selector'].ranking_
    print(f"Optimal number of features : {selector.n_features_}")
    # Get the feature names after preprocessing
    f_names = get_features_after_onehot(pipe_selector)
    for i in range(len(f_names)):
        if support[i]:
            print(f"{f_names[i]}: {ranking[i]}")
            features.append(f_names[i])
    print()
    for feat in features: feature_counts[feat] += 1
    return selector, features

def transform_and_select(X, preprocessor, features, fit=True):
    p = Pipeline(steps=[('preprocessor', preprocessor)])
    preproc = p.fit(X) if fit else p
    np_X = preproc.transform(X)
    f_names = get_features_after_onehot(p)
    X_df = pd.DataFrame(data=np_X, columns=f_names)
    return preproc.named_steps['preprocessor'], X_df[features]

def cvscore(key, X, y, clf, preprocessor, scoring, cv, selector=None):
    if selector:
        sel_fit, selected_f = feature_selection(X, y, preprocessor, selector)
        pre, X_fs = transform_and_select(X, preprocessor, selected_f)
        selected_features.update({key: selected_f})
        pipe = Pipeline(steps=[('classifier', clf)])
        cvscores = cross_validate(pipe, X_fs, y, scoring=scoring, cv=cv,
                n_jobs=-1)
    else:
        pipe = Pipeline(steps=[('preproc', preprocessor), ('classifier', clf)])
        cvscores = cross_validate(pipe, X, y, scoring=scoring, cv=cv,
                n_jobs=-1)
    s = print_scores(key, clf, scoring, cvscores)
    final_scores.update(s)

def cross_validation_scores(key, clf, X, y, scoring, cv):
    print(f"Cross Validation for: {key}")
    cvscore(key, X, y, clf, preprocessor, scoring, cv)
    print(f"Cross Validation for: {key}; Standard Scaler")
    cvscore(f"{key}_std", X, y, clf, preprocessor_std, scoring, cv)
    print(f"Cross Validation for: {key}; Power Transformer")
    cvscore(f"{key}_pt", X, y, clf, preprocessor_pt, scoring, cv)
    # Feature selection
    selector = RFECV(clf, step=1, cv=cv, n_jobs=-1)
    print(f"Cross Validation for: {key}; RFE")
    cvscore(f"{key}_fs", X, y, clf, preprocessor, scoring, cv, selector)
    print(f"Cross Validation for: {key}; RFE with Standard Scaler")
    cvscore(f"{key}_fs_std", X, y, clf, preprocessor_std, scoring, cv, selector)
    print(f"Cross Validation for: {key}; RFE with Power Transformer")
    cvscore(f"{key}_fs_pt", X, y, clf, preprocessor_pt, scoring, cv, selector)

def get_features_after_onehot(pipe, strip=True):
    # get_feature_names() change feature names, but OneHot Encoding
    # transformation filter don't
    f_names_cat = pipe.named_steps['preprocessor'].\
            named_transformers_['cat'].named_steps['onehot'].\
            get_feature_names(input_features=categoricals).tolist()
    if strip:
        return numerical + [f.replace('type_', '') for f in f_names_cat]
    else:
        return numerical + f_names_cat

def univariate_feature_selection(X, y, preprocessor, score_func, perc):
    selector = SelectPercentile(score_func, percentile=perc)
    pipe_selector = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector)])
    pipe_selector.fit(X, y)
    features = get_features_after_onehot(pipe_selector)
    selected_features.update({'all': features})
    s_dict = {features[i]: selector.scores_[i] for i in range(len(features))}
    if type(selector.pvalues_) != type(None):
        scores = [(features[i], selector.scores_[i], selector.pvalues_[i])
                for i in range(len(features)) if selector.get_support()[i]]
    else:
        scores = [(features[i], selector.scores_[i])
                for i in range(len(features)) if selector.get_support()[i]]
    ord_scores = sorted(scores, key=lambda k: k[1], reverse=True)
    for s in ord_scores:
        p = f"{s[2]:0.5f}" if len(s)>2 else None
        print(f"{s[0]}\tF-value: {s[1]:0.5f}\tp-value: {p}")
    return s_dict

print(f"Univariate Feature Selection")
print("========================================")

print(f"ANOVA test with data standardization (all features)")
f_scores = univariate_feature_selection(X_train, y_train, preprocessor_std,
        f_classif, 100)
print("----------------------------------------")
print(f"ANOVA test with data standardization (10% best features)")
univariate_feature_selection(X_train, y_train, preprocessor_std, f_classif, 10)
print("----------------------------------------")

print(f"Mutual information with plain data (all features)")
mi_scores = univariate_feature_selection(X_train, y_train, preprocessor,
        mutual_info_classif, 100)
print("----------------------------------------")
print(f"Mutual information with plain data (10% best features)")
univariate_feature_selection(X_train, y_train, preprocessor,
        mutual_info_classif, 10)
print("----------------------------------------")

print("========================================")
print(f"Cross Validation results")
print("========================================")

## Decision Tree
clf = DecisionTreeClassifier(random_state=0)
cross_validation_scores('dt', clf, X_train, y_train, scoring, skf)

### Gradient Boosting
clf = GradientBoostingClassifier(n_estimators=500, random_state=0)
cross_validation_scores('gb', clf, X_train, y_train, scoring, skf)

## Random Forests
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
cross_validation_scores('rf', clf, X_train, y_train, scoring, skf)

# Print table of scores
print("========================================")
headers = f"Classifier"
headers_complete = False
table = ""
for key, scores_dict in final_scores.items():
    table += f"\n{key}"
    for score, scores in scores_dict.items():
        if not headers_complete:
            headers += f"\t{score} mean (+/- error)"
        table += f"\t{scores.mean():0.4f} ({scores.std()*2:0.4f})"
    headers_complete = True
print(headers)
print(table)
print("========================================")
print("Feature\tF Score\tMI Score\tHits")
feats = [(f, c, f_scores[f], mi_scores[f]) for f, c in feature_counts.items()]
for (f, c, fs, ms) in sorted(feats, key=lambda x: (x[1], x[3]), reverse=True):
    print(f"{f}\t{fs:0.4f}\t{ms:0.4f}\t{c}")
print("========================================")

# Fit the models
test_scores = {}

def dump_tree(clf, feature_names=None, fname='image.pdf'):
    from sklearn.externals.six import StringIO
    from sklearn.tree import export_graphviz
    import pydot
    dot_data = StringIO()
    export_graphviz(clf, class_names=['exchange', 'non-exchange'],
            feature_names=feature_names, filled=True, rounded=True,
            out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf(fname)

def print_cm(y_test, y_pred, labels):
    print(f"\nConfussion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    print(f"{labels}\n{cm}")
    return tn, fp, fn, tp

def print_wrong_classified(X_test, y_test, y_pred):
    print(f"\nWrong classified instances:")
    bad_pred = defaultdict(set)
    for idx in y_test[y_test!=y_pred].index:
        a = X_test.loc[idx]['address']
        l = y_test.loc[idx]
        bad_pred[l].add(a)
        print(f"[{idx}]\t{a}\t{l}")
    return bad_pred

def print_feature_importances(clf, f_names):
    print(f"\nFeature importances:")
    tt = [(f, clf.feature_importances_[i]) for i, f in enumerate(f_names)]
    # Sort by importance
    for (f, i) in sorted(tt, key=lambda k: k[1]):
        print(f'{f}: {i:.5f}')
    print("----------------------------------------")

def model_prediction(key, X_train, y_train, X_test, y_test, clf, preproc,
        features=None):
    if features:
        # TODO check if it is possible to drop unused features instead
        fitted_prep, X_train_ = transform_and_select(X_train, preproc, features)
        prep, X_test_ = transform_and_select(X_test, fitted_prep, features, fit=False)
        pipe = Pipeline(steps=[('classifier', clf)])
    else:
        X_train_ = X_train
        X_test_ = X_test
        pipe = Pipeline(steps=[('preprocessor', preproc), ('classifier', clf)])
    pipe.fit(X_train_, y_train)
    y_pred = pipe.predict(X_test_)
    score = pipe.score(X_test_, y_test)
    # Get the feature names after OneHot Encoding
    if features:
        f_names = features
    else:
        f_names = get_features_after_onehot(pipe, strip=False)
    tn, fp, fn, tp = print_cm(y_test, y_pred, labels)
    p = precision_score(y_test, y_pred, pos_label='exchange', labels=labels)
    r = recall_score(y_test, y_pred, pos_label='exchange', labels=labels)
    f = f1_score(y_test, y_pred, pos_label='exchange', labels=labels)
    print('')
    print(f"\nModel:\tAccuracy:\tPrecision:\tRecall:\tF1:")
    print(f"{key}\t{score:.4f}\t{p:.4f}\t{r:.4f}\t{f:.4f}")
    test_scores.update({key: {'a': score, 'p': p, 'r': r, 'f': f}})
    wc = print_wrong_classified(X_test, y_test, y_pred)
    for l, addrs in wc.items():
        for addr in addrs:
            wrong_counts[l][addr] += 1
    print_feature_importances(clf, f_names)
    dump(pipe, f"{key}_classifier.joblib")

def train_test_model(key, clf):
    print(f">> {key} No standardization")
    model_prediction(key, X_train, y_train, X_test, y_test, clf, preprocessor)
    print(f">> {key} Standard Scaler (z-score)")
    model_prediction(f'{key}_std', X_train, y_train, X_test, y_test, clf, preprocessor_std)
    print(f">> {key} Power Transformer")
    model_prediction(f'{key}_pt', X_train, y_train, X_test, y_test, clf, preprocessor_pt)
    print(f">> {key} Feature Selection, No standardization")
    model_prediction(f'{key}_fs', X_train, y_train, X_test, y_test, clf, preprocessor,
            selected_features[f"{key}_fs"])
    print(f">> {key} Feature Selection, Standard Scaler (z-score)")
    model_prediction(f'{key}_fs_std', X_train, y_train, X_test, y_test, clf, preprocessor_std,
            selected_features[f"{key}_fs_std"])
    print(f">> {key} Feature Selection, Power Transformer")
    model_prediction(f'{key}_fs_pt', X_train, y_train, X_test, y_test, clf, preprocessor_pt,
            selected_features[f"{key}_fs_pt"])

def grid_search(clf, preprocessor, param_grid, name):
    # Grid search
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', clf)])
    grid = GridSearchCV(pipe, cv=skf, param_grid=param_grid,
            scoring='f1_macro')
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    estimator = grid.best_estimator_
    print(f"Best Estimator: {estimator}")
    print(f"Best Score: {grid.best_score_}")
    print(f"mean_test_score: {grid.cv_results_['mean_test_score']}")

    # Last prediction
    y_pred = estimator.predict(X_test)
    score = estimator.score(X_test, y_test)
    p = precision_score(y_test, y_pred, pos_label='exchange', labels=labels)
    r = recall_score(y_test, y_pred, pos_label='exchange', labels=labels)
    f = f1_score(y_test, y_pred, pos_label='exchange', labels=labels)
    print('\nGrid-search scores:')
    print(f"Accuracy: {score:.4f}\tP: {p:.4f}\tR: {r:.4f}\tF1: {f:.4f}")
    tn, fp, fn, tp = print_cm(y_test, y_pred, labels)
    wc = print_wrong_classified(X_test, y_test, y_pred)
    print(f"Exporting model tests")
    dump(estimator, name)

    fitted_est = load(name)
    fitted_y = fitted_est.predict(X_test)
    fitted_score = fitted_est.score(X_test, y_test)
    print(f"Exported model score: {fitted_score}")
    tn, fp, fn, tp = print_cm(y_test, fitted_y, labels)
    wc = print_wrong_classified(X_test, y_test, fitted_y)

print(f"Model results")
print("========================================")

wrong_counts = {'exchange': defaultdict(int), 'non-exchange': defaultdict(int)}

print(">> Decision Tree [Train:80%, Test:20%]")
clf = DecisionTreeClassifier(random_state=0)
train_test_model('dt', clf)

print(">> Gradient Boosting [Train:80%, Test:20%]")
clf = GradientBoostingClassifier(n_estimators=500, random_state=0)
train_test_model('gb', clf)

print(">> Random Forests [Train:80%, Test:20%]")
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
train_test_model('rf', clf)

# Print table of test scores
print("========================================")
headers = f"Classifier"
headers_complete = False
table = ""
for key, scores in test_scores.items():
    table += f"\n{key}"
    for score, value in scores.items():
        if not headers_complete:
            headers += f"\t{score}"
        table += f"\t{value:0.4f}"
    headers_complete = True
print(headers+table)
print("========================================")

# Grid search on the best scoring pipeline
clf = RandomForestClassifier(random_state=0)
param_grid = {
    'classifier__n_estimators': [400, 500, 600, 700, 800, 900, 1000],
    'classifier__bootstrap': [True, False],
    'classifier__max_depth': [10, 20, 30, 40, 50, None],
    'classifier__max_features': ['sqrt', 'log2', 0.1, 0.3, 0.4],
    'classifier__min_samples_leaf': [1, 2, 3, 4],
    'classifier__min_samples_split': [2, 4, 6, 10],
    }
grid_search(clf, preprocessor_std, param_grid, 'classifier_gridsearch.joblib')


# Print address wrong classified
print("========================================")
print("Frequency of wrong classified addresses:")
for l, d in wrong_counts.items():
    for addr, count in sorted(d.items(), key=lambda x: x[1], reverse=True):
        print(f"{l}\t{addr}\t{count}")

