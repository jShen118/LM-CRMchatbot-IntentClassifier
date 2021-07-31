'''

'''

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True, ngram_range=(1, 2))
#vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
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
labelsList = ['Intent.AccessIssues'] * len(accessNormalized)
labelsList += ['Intent.CallQualityIssues'] * len(callqualityNormalized)
labelsList += ['Intent.FrozenLoadingIssue'] * len(frozenloadingNormalized)
labelsList += ['Intent.GRMIssues'] * len(grmNormalized)
labelsList += ['Intent.GRSIssues'] * len(grsNormalized)
labelsList += ['Intent.MobileManagement'] * len(mobilemanagementNormalized)
labelsList += ['Intent.NetworkIssues'] * len(networkNormalized)
labelsList += ['Intent.OutlookIssues'] * len(outlookNormalized)
labelsList += ['Intent.RatingIssues'] * len(ratingNormalized)
labelsList += ['Intent.HardWareIssues'] * len(hardwareNormalized)
labelsList += ['None'] * len(noneNormalized)

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
        return (classifier_linear.predict(utterance_vector)[0], max(classifier_linear.predict_proba(utterance_vector)[0]))
    elif kernel == 'poly':
        return (classifier_poly.predict(utterance_vector)[0], max(classifier_poly.predict_proba(utterance_vector)[0]))
    elif kernel == 'rbf':
        return (classifier_rbf.predict(utterance_vector)[0], max(classifier_rbf.predict_proba(utterance_vector)[0]))
    elif kernel == 'sigmoid':
        return (classifier_sigmoid.predict(utterance_vector)[0], max(classifier_sigmoid.predict_proba(utterance_vector)[0]))
    return None

def SVMclassifyUtterances(utterances, kernel):
    return [(u, SVMclassify(u, kernel)) for u in utterances]

#param [(<comment>, (<label>, <confidence>))]
def printLabels(labelledUtterances):
    print('OUTPUT OF SUPPORT VECTOR MACHINE CLASSIFIER:\n\n')
    for lu in labelledUtterances:
        print(lu[1][0].upper(), round(lu[1][1], 3), ':', f'\"{lu[0]}\"', '\n')


