from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

clf = KNeighborsClassifier(n_neighbors=9)
clfrf = RandomForestClassifier()

scores = cross_val_score(clf, sentvecs, sentiments, cv=5)
print((np.mean(scores), np.std(scores)))

scores = cross_val_score(clfrf, sentvecs, sentiments, cv=5)
print((np.mean(scores), np.std(scores)))

# bag-of-words comparison
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), RandomForestClassifier())
scores = cross_val_score(pipeline, sentences, sentiments, cv=5)
print((np.mean(scores), np.std(scores)))
