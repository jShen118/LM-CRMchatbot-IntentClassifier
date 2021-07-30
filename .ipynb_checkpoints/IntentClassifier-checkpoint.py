'''

'''

from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import readjsondata as data



#TRAINING DATA PREPARATION
accessUtterances = data.AccessUtterances
callqualityUtterances = data.CallQualityUtterances
frozenloadingUtterances = data.FrozenLoadingUtterances
grmUtterances = data.GRMUtterances
grsUtterances = data.GRSUtterances
mobilemanagamentUtterances = data.MobileManagementUtterances
networkUtterances = data.NetworkUtterances
outlookUtterances = data.OutlookUtterances
ratingUtterances = data.RatingUtterances
hardwareUtterances = data.HardwareUtterances
noneUtterances = data.NoneUtterances

sw = set(stopwords.words('english') + list(punctuation))
sw.union(['trouble', 'having'])

def removeStopwords(utterance):
    #removes stopwords from utterances
    return ' '.join([word for word in utterance.split() if word.lower() not in sw])
def removeStopwordsList(utterances):
    return [u for u in list(map(removeStopwords, utterances)) if u]

#param: list<utterance>
#return: list<words>
def bagOfWords(utterances):
    utterances = list(filter(lambda s: s , list(map(removeStopwords, utterances))))
    return [word.lower() for u in utterances for word in u.split()]

accessWords = bagOfWords(accessUtterances)
callqualityWords = bagOfWords(accessUtterances)
frozenloadingWords = bagOfWords(accessUtterances)
grmWords = bagOfWords(accessUtterances)
grsWords = bagOfWords(accessUtterances)
mobilemanagementWords = bagOfWords(accessUtterances)
networkWords = bagOfWords(accessUtterances)
outlookWords = bagOfWords(accessUtterances)
ratingWords = bagOfWords(accessUtterances)
hardwareWords = bagOfWords(accessUtterances)
noneWords = bagOfWords(accessUtterances)

accessUtterances = removeStopwordsList(accessUtterances)
callqualityUtterances = removeStopwordsList(accessUtterances)
frozenloadingUtterances = removeStopwordsList(accessUtterances)
grmUtterances = removeStopwordsList(accessUtterances)
grsUtterances = removeStopwordsList(accessUtterances)
mobilemanagementUtterances = removeStopwordsList(accessUtterances)
networkUtterances = removeStopwordsList(accessUtterances)
outlookUtterances = removeStopwordsList(accessUtterances)
ratingUtterances = removeStopwordsList(accessUtterances)
hardwareUtterances = removeStopwordsList(accessUtterances)
noneUtterances = removeStopwordsList(accessUtterances)

vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
train_vectors = vectorizer.fit_transform(
    accessUtterances + 
    callqualityUtterances + 
    frozenloadingUtterances + 
    grmUtterances + 
    grsUtterances + 
    mobilemanagementUtterances + 
    networkUtterances + 
    outlookUtterances + 
    ratingUtterances + 
    hardwareUtterances + 
    noneUtterances
)
labelsList = ['Intent.AccessIssues'] * len(accessUtterances)
labelsList += ['Intent.CallQualityIssues'] * len(callqualityUtterances)
labelsList += ['Intent.FrozenLoadingIssue'] * len(frozenloadingUtterances)
labelsList += ['Intent.GRMIssues'] * len(grmUtterances)
labelsList += ['Intent.GRSIssues'] * len(grsUtterances)
labelsList += ['Intent.MobileManagement'] * len(mobilemanagementUtterances)
labelsList += ['Intent.NetworkIssues'] * len(networkUtterances)
labelsList += ['Intent.OutlookIssues'] * len(outlookUtterances)
labelsList += ['Intent.RatingIssues'] * len(ratingUtterances)
labelsList += ['Intent.HardWareIssues'] * len(hardwareUtterances)
labelsList += ['None'] * len(noneUtterances)

classifier_linear = SVC(kernel='linear', probability = True)
classifier_linear.fit(train_vectors, labelsList)
classifier_poly = SVC(kernel='poly', probability = True)
classifier_poly.fit(train_vectors, labelsList)
classifier_rbf = SVC(kernel='rbf', probability = True)
classifier_rbf.fit(train_vectors, labelsList)
classifier_sigmoid = SVC(kernel='sigmoid', probability = True)
classifier_sigmoid.fit(train_vectors, labelsList)

# svm kernel can be ‘linear’, ‘poly’, ‘rbf’, or ‘sigmoid’
def SVMclassify(utterance, kernel):
    utterance = removeStopwords(utterance)
    utterance_vector = vectorizer.transform([utterance]) # vectorizing
    if kernel == 'linear':
        return (classifier_linear.predict(utterance_vector)[0], max(classifier_rbf.predict_proba(utterance_vector)[0]))
    elif kernel == 'poly':
        return (classifier_poly.predict(utterance_vector)[0], max(classifier_rbf.predict_proba(utterance_vector)[0]))
    elif kernel == 'rbf':
        return (classifier_rbf.predict(utterance_vector)[0], max(classifier_rbf.predict_proba(utterance_vector)[0]))
    elif kernel == 'sigmoid':
        return (classifier_sigmoid.predict(utterance_vector)[0], max(classifier_rbf.predict_proba(utterance_vector)[0]))
    return None

def SVMclassifyUtterances(utterances, kernel):
    return [(u, SVMclassify(u, kernel)) for u in utterances]

#param [(<comment>, (<label>, <confidence>))]
def printLabels(labelledUtterances):
    print('OUTPUT OF SUPPORT VECTOR MACHINE CLASSIFIER:\n\n')
    for lu in labelledUtterances:
        print(lu[1][0].upper(), round(lu[1][1], 3), ':', f'\"{lu[0]}\"', '\n')


