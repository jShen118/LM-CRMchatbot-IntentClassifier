'''

'''

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.svm import SVC
import rwjson

#utterance normalization functions
def removePunctuation(utterance):
    return utterance.translate(str.maketrans('', '', punctuation))
def removeStopwords(utterance):
    sw = set(stopwords.words('english')) #sw.union(['trouble', 'having'])
    return ' '.join([word for word in utterance.split() if word.lower() not in sw])
def stem(utterance):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in utterance.split()])

#stem made all three batch tests much less accurate (>20% or so depending)
#removeStopwords and removePunctuation make 75-1 and 75-2 a few pts more accurate, but 75-3 very slightly less accurate
#so for now normalize will be removeStopwords and removePunctuation, no stemming
def normalize(utterance):
    return removeStopwords(removePunctuation(utterance)).lower()
def normalizeUtterances(utterances):
    return list(map(normalize, utterances))


data = rwjson.readjson()

accessNormalized = normalizeUtterances(data.access)
callqualityNormalized = normalizeUtterances(data.callquality)
frozenloadingNormalized = normalizeUtterances(data.frozenloading)
grmNormalized = normalizeUtterances(data.grm)
grsNormalized = normalizeUtterances(data.grs)
mobilemanagementNormalized = normalizeUtterances(data.mobilemanagement)
networkNormalized = normalizeUtterances(data.network)
outlookNormalized = normalizeUtterances(data.outlook)
ratingNormalized = normalizeUtterances(data.rating)
hardwareNormalized = normalizeUtterances(data.hardware)
noneNormalized = normalizeUtterances(data.none)


#Training Support Vector Machines
#vectorizer = CountVectorizer(min_df = 3, max_df = 0.25, ngram_range=(1, 2))
vectorizer = TfidfVectorizer(min_df = 3, max_df = 0.25, sublinear_tf = True, ngram_range=(1, 3))
#vectorizer = HashingVectorizer(ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(
    accessNormalized + 
    callqualityNormalized + 
    frozenloadingNormalized + 
    grmNormalized + 
    grsNormalized + 
    mobilemanagementNormalized + 
    networkNormalized + 
    outlookNormalized + 
    ratingNormalized + 
    hardwareNormalized + 
    noneNormalized
)
intents = [
        'Intent.AccessIssues',
        'Intent.CallQualityIssues',
        'Intent.FrozenLoadingIssue',
        'Intent.GRMIssues',
        'Intent.GRSIssues',
        'Intent.MobileManagement',
        'Intent.NetworkIssues',
        'Intent.OutlookIssues',
        'Intent.RatingIssues',
        'Intent.HardWareIssues',
        'None'
    ]
labelsList = [intents[0]] * len(accessNormalized)
labelsList += [intents[1]] * len(callqualityNormalized)
labelsList += [intents[2]] * len(frozenloadingNormalized)
labelsList += [intents[3]] * len(grmNormalized)
labelsList += [intents[4]] * len(grsNormalized)
labelsList += [intents[5]] * len(mobilemanagementNormalized)
labelsList += [intents[6]] * len(networkNormalized)
labelsList += [intents[7]] * len(outlookNormalized)
labelsList += [intents[8]] * len(ratingNormalized)
labelsList += [intents[9]] * len(hardwareNormalized)
labelsList += [intents[10]] * len(noneNormalized)

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.fit
C = 1 #penalty for missclassified vectors
classifier_linear = SVC(kernel='linear', probability = True, C=C)
classifier_linear.fit(train_vectors, labelsList)
classifier_poly = SVC(kernel='poly', probability = True, C=C)
classifier_poly.fit(train_vectors, labelsList)
classifier_rbf = SVC(kernel='rbf', probability = True, C=C)
classifier_rbf.fit(train_vectors, labelsList)
classifier_sigmoid = SVC(kernel='sigmoid', probability = True, C=C)
classifier_sigmoid.fit(train_vectors, labelsList)

# svm kernel can be ‘linear’, ‘poly’, ‘rbf’, or ‘sigmoid’
#returns (<intent>, list<(<probability>, <intent>)>)
def SVMpredict(utterance, kernel='rbf'):
    utterance_vector = vectorizer.transform([normalize(utterance)])
        
    if kernel == 'linear':
        probabilities = classifier_linear.predict_proba(utterance_vector)[0]
        return (classifier_linear.predict(utterance_vector)[0], sorted(zip(probabilities, intents), reverse = True))
    elif kernel == 'poly':
        probabilities = classifier_poly.predict_proba(utterance_vector)[0]
        return (classifier_poly.predict(utterance_vector)[0], sorted(zip(probabilities, intents), reverse = True))
    elif kernel == 'rbf':
        probabilities = classifier_rbf.predict_proba(utterance_vector)[0]
        return (classifier_rbf.predict(utterance_vector)[0], sorted(zip(probabilities, intents), reverse = True))
    elif kernel == 'sigmoid':
        probabilities = classifier_sigmoid.predict_proba(utterance_vector)[0]
        return (classifier_sigmoid.predict(utterance_vector)[0], sorted(zip(probabilities, intents), reverse = True))
    return None

def SVMpredictUtterances(utterances, kernel):
    return [(u, SVMpredict(u, kernel)) for u in utterances]

#predictions is (<intent>, list<(<probability>, <intent>)>)
#returns (<intent>, <probability>)
def classify(predictions):
    #if predictions[0] != predictions[1][0][1]: print('mismatch')
    #first for loop is for None override condition
    #second for loop is for finding the classifier.predict probability
        #unfortunately the top prediction probability does not always match with the predict return
        #has something to do with Platt scaling https://ronie.medium.com/sklearn-svc-predict-vs-predict-proba-e594293153c1
        #takeaway is that probability in the case of mismatch is less important than predict result
    for p in predictions[1]:
        if p[1] == 'None' and p[0] > 0.11:
            return ('None', p[0])
    for p in predictions[1]:
        if p[1] == predictions[0]:
            return (predictions[0], p[0])
    print('THIS SHOULD NEVER EXECUTE')
    return None
