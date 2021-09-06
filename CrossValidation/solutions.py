########## Exercise: implementing stratified K-fold CV

predictions = []
targets = []

for train, test in splitter.split(df,y = df['species']):
    forest = RandomForestClassifier(trees, n_jobs=jobs)
    forest.fit(df.iloc[train][features],df.iloc[train]['species'])
    
    results = forest.predict(df.iloc[test][features])
    targets += list(df.iloc[test]['species'])
    predictions += list(results)

test_accuracy = accuracy_score(predictions,targets)
print('Test accuracy',test_accuracy)









########## Exercise: nested stratified K-fold CV
parameters = {'n_estimators': [10,100,1000], 'max_depth': (1,2,None)}
outer = StratifiedKFold(10,shuffle=True)
inner = StratifiedKFold(10,shuffle=True)

predictions = []
targets = []

import tqdm
for train, test in tqdm.tqdm(outer.split(df,y = df['species']),total = 10):
    gs = GridSearchCV(estimator=RandomForestClassifier(n_jobs=jobs),
            param_grid=parameters,
            cv = inner,
            verbose = 0
            )
    gs.fit(df.iloc[train][features],df.iloc[train]['species'])
    
    # by default, the best classifier is retrained on the whole data
    
    results = gs.predict(df.iloc[test][features])
    targets += list(df.iloc[test]['species'])
    predictions += list(results)

test_accuracy = accuracy_score(predictions,targets)
print('Test accuracy',test_accuracy)
