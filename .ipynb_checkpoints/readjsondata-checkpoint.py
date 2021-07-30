'''
This is a python script to read data from training.json to split utterances by their intents into separate lists.
There should be 411 total utterances that span over 11 intents (including None):
    Intent.AccessIssues
    Intent.CallQualityIssues
    Intent.FrozenLoadingIssue
    Intent.GRMIssues
    Intent.GRSIssues
    Intent.MobileManagement
    Intent.NetworkIssues
    Intent.OutlookIssues
    Intent.RatingIssues
    Intent.HardWareIssues
    None
'''

import json
print('hey')

AccessUtterances = []
CallQualityUtterances = []
FrozenLoadingUtterances = []
GRMUtterances = []
GRSUtterances = []
MobileManagementUtterances = []
NetworkUtterances = []
OutlookUtterances = []
RatingUtterances = []
HardwareUtterances = []
NoneUtterances = []

f = open('training.json')
data = json.load(f)
for d in data:
    intent = d['intentName']
    utterance = d['text']
    if intent == 'Intent.AccessIssues':
        AccessUtterances.append(utterance)
    elif intent == 'Intent.CallQualityIssues':
        CallQualityUtterances.append(utterance)
    elif intent == 'Intent.FrozenLoadingIssue':
        FrozenLoadingUtterances.append(utterance)
    elif intent == 'Intent.GRMIssues':
        GRMUtterances.append(utterance)
    elif intent == 'Intent.GRSIssues':
        GRSUtterances.append(utterance)
    elif intent == 'Intent.MobileManagement':
        MobileManagementUtterances.append(utterance)
    elif intent == 'Intent.NetworkIssues':
        NetworkUtterances.append(utterance)
    elif intent == 'Intent.OutlookIssues':
        OutlookUtterances.append(utterance)
    elif intent == 'Intent.RatingIssues':
        RatingUtterances.append(utterance)
    elif intent == 'Intent.HardWareIssues':
        HardwareUtterances.append(utterance)
    elif intent == 'None':
        NoneUtterances.append(utterance)
    else:
        print(f'Something is wrong with training.json. Mislabeled intent. {intent}')

def printUtterances(utterances, intentName):
    toPrint = f'{len(utterances)} {intentName} Utterances:\n'
    for u in utterances:
        toPrint += f'{u}\n'
    print(toPrint)
    
'''
printUtterances(AccessUtterances, 'Intent.AccessIssues')
printUtterances(CallQualityUtterances, 'Intent.CallQualityIssues')
printUtterances(FrozenLoadingUtterances, 'Intent.FrozenLoadingIssue')
printUtterances(GRMUtterances, 'Intent.GRMIssues')
printUtterances(GRSUtterances, 'Intent.GRSIssues')
printUtterances(MobileManagementUtterances, 'Intent.MobileManagement')
printUtterances(NetworkUtterances, 'Intent.NetworkIssues')
printUtterances(OutlookUtterances, 'Intent.OutlookIssues')
printUtterances(RatingUtterances, 'Intent.RatingIssues')
printUtterances(HardwareUtterances, 'Intent.HardWareIssues')
printUtterances(NoneUtterances, 'None')
'''
