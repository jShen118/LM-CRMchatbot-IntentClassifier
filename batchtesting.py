'''

'''
import json
import pandas as pd
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
def avg(ls):
    toRet = 0
    for e in ls:
        if e is not None:
            toRet += e
    toRet /= len(ls)
    return round(toRet, 3)
#returns list fScores in order of standard intent order, last element is avg
def fScores(truePos, falsePos, falseNeg):
    #harmonic mean of precision and recall https://en.wikipedia.org/wiki/F-score
    #print(truePos, falsePos, falseNeg)
    def fScore(precision, recall):
        return round(2/(1/precision + 1/recall), 3)
    fscores = []
    for intent in ic.intents:
        if truePos[intent] == 0:
            fscores.append(None)
        else:
            precision = truePos[intent]/(truePos[intent] + falsePos[intent])
            recall = truePos[intent]/(truePos[intent] + falseNeg[intent])
            fscore = fScore(precision, recall)
            fscores.append(fscore)
    fscores.append(avg(fscores))
    return fscores


def batchtest(file, verbose):
    f = open(file)
    data = json.load(f)
    f.close()
    correct = []
    incorrect = []

    # precision: truePos/truePos+falsePos, recall: truePos/truePos+falseNeg
    truePos = intentsDict()
    falsePos = intentsDict()
    falseNeg = intentsDict()
    
    for d in data:
        label = ic.classify(ic.SVMpredict(d['text']))
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
    
    toPrint = f'BATCH TEST OF {file}\n'
    toPrint += f'{len(data)} Utterances. {len(correct)}/{total} {round(100*len(correct)/total, 2)}% Labeled Correctly\n'
    
    toPrint += 'F-Scores:\n'
    fscores = fScores(truePos, falsePos, falseNeg)
    for i in range(len(fscores)-1):
        if fscores[i] is None:
            toPrint += f'  {ic.intents[i]}: N/A (true positive of 0, cannot divide by 0)\n'
        else:
            toPrint += f'  {ic.intents[i]}: {fscores[i]}\n'
    toPrint += f'  Average: {fscores[-1]}\n'
    
    
    if verbose:
        toPrint += 'CORRECT:\n'
        for lu in correct:
            toPrint += '  ' + lu + '\n'
        toPrint += 'INCORRECT:\n'
        for lu in incorrect:
            toPrint += '  ' + lu + '\n'
    toPrint += '\n\n'
    print(toPrint)

def runBatchtests(files):
    jsondata = [json.load(open(f)) for f in files]
    dfdata = []
    shorthandIntents = [
        'Access',
        'Call',
        'Frozen',
        'GRM',
        'GRS',
        'Mobile',
        'Network',
        'Outlook',
        'Rating',
        'Hardware',
        'None'
    ]
    def shortenPath(path):
        return path[path.index('/')+1:path.index('.')]
    for bt in jsondata:
        truePos = intentsDict()
        falsePos = intentsDict()
        falseNeg = intentsDict()
        for d in bt:
            label = ic.classify(ic.SVMpredict(d['text']))
            if label[0] == d['intent']:
                truePos[label[0]] += 1
            else:
                falsePos[label[0]] += 1
                falseNeg[d['intent']] += 1
        dfdata.append(fScores(truePos, falsePos, falseNeg))
    avgRow = [0] * 12
    for i in range(0, 12):
        for fscores in dfdata:
            if fscores[i] is not None:
                avgRow[i] += fscores[i]
        avgRow[i] = round(avgRow[i]/len(dfdata), 3)
    dfdata.append(avgRow)
    df = pd.DataFrame(dfdata, columns=shorthandIntents+['Avg'], index=[shortenPath(f) for f in files]+['Avg'])
    display(df)
    df.plot(kind='bar', figsize=(13,9), xlabel='Batch Test', ylabel='F-Score')





