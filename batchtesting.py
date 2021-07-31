'''

'''

import json
import IntentClassifier as ic



def centerBuffered(string, spaces):
    #i'll get to this later, it's only to make verbose output prettier
    return ' '
#produces dictionary of <intent: 0> for all 11 intents
def intentsDict():
    return {
        'Intent.AccessIssues': 0,
        'Intent.CallQualityIssues': 0,
        'Intent.FrozenLoadingIssue': 0,
        'Intent.GRMIssues': 0,
        'Intent.GRSIssues': 0,
        'Intent.MobileManagement': 0,
        'Intent.NetworkIssues': 0,
        'Intent.OutlookIssues': 0,
        'Intent.RatingIssues': 0,
        'Intent.HardWareIssues': 0,
        'None': 0
    }
def printFScores(truePos, falsePos, falseNeg):
    #harmonic mean of precision and recall https://en.wikipedia.org/wiki/F-score
    #print(truePos, falsePos, falseNeg)
    def fScore(precision, recall):
        return round(2/(1/precision + 1/recall), 3)
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
    toPrint = 'F-Scores:\n'
    fscores = []
    for intent in intents:
        if truePos[intent] == 0:
            toPrint += f'  {intent}: N/A (true positive of 0, cannot divide by 0)\n'
        else:
            precision = truePos[intent]/(truePos[intent] + falsePos[intent])
            recall = truePos[intent]/(truePos[intent] + falseNeg[intent])
            fscore = fScore(precision, recall)
            fscores.append(fscore)
            toPrint += f'  {intent}: {fscore}\n'
    avg = 0
    for s in fscores:
        avg += s
    avg /= len(fscores)
    toPrint += f'  Average: {avg}'
    print(toPrint)


def batchtest(file, kernel, verbose):
    f = open(file)
    data = json.load(f)
    correct = []
    incorrect = []

    # precision: truePos/truePos+falsePos, recall: truePos/truePos+falseNeg
    truePos = intentsDict()
    falsePos = intentsDict()
    falseNeg = intentsDict()
    
    for d in data:
        label = ic.classify(ic.SVMpredict(d['text'], kernel))
        labelledUtterance = f'Labeled {label[0]} {round(label[1], 3)} | Actually ' + d['intent'] + ' | ' + d['text']
        if label[0] == d['intent']:
            #for each right classification there's a true positive (and a true negative but that doesn't factor into f-score)
            truePos[label[0]] += 1
            correct.append(labelledUtterance)
        else:
            #for each misclassification there is a false positive and a false negative
            falsePos[label[0]] += 1
            falseNeg[d['intent']] += 1
            incorrect.append(labelledUtterance)
    total = len(correct) + len(incorrect)
    
    print(f'BATCH TEST OF {file}')
    print(f'{len(data)} Utterances. {len(correct)}/{total} {round(100*len(correct)/total, 2)}% Labeled Correctly')
    
    printFScores(truePos, falsePos, falseNeg)
    
    if verbose:
        print('CORRECT:')
        for lu in correct:
            print('  ' + lu)
        print('INCORRECT:')
        for lu in incorrect:
            print('  ' + lu)
    print('\n\n')









