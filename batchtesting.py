'''

'''

import json
import IntentClassifier as ic

def batchtest(file, kernel, verbose):
    f = open(file)
    data = json.load(f)
    correct = []
    incorrect = []
    for d in data:
        label = ic.SVMclassify(d['text'], kernel)
        labelledUtterance = f'Labeled {label[0]} {round(label[1], 3)} | Actually ' + d['intent'] + ' | ' + d['text']
        #labelledUtterance = f'Labeled {label[0]} {label[1]} '
        if label[0] == d['intent']:
            correct.append(labelledUtterance)
        else:
            incorrect.append(labelledUtterance)
    total = len(correct) + len(incorrect)
    print(f'{len(data)} Utterances. {len(correct)}/{total} {round(100*len(correct)/total, 2)}% Accuracy:\n')
    if verbose:
        print('CORRECT:')
        for lu in correct:
            print(lu)
        print('\nINCORRECT:')
        for lu in incorrect:
            print(lu)
        print('\n\n\n')
            









