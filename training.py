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

#reads from training.json and returns Utterances object
def readjson():
    class Utterances:
        access = []
        callquality = []
        frozenloading = []
        grm = []
        grs = []
        mobilemanagement = []
        network = []
        outlook = []
        rating = []
        hardware = []
        none = []

    f = open('training.json')
    data = json.load(f)
    for d in data:
        intent = d['intentName']
        utterance = d['text']
        if intent == 'Intent.AccessIssues':
            Utterances.access.append(utterance)
        elif intent == 'Intent.CallQualityIssues':
            Utterances.callquality.append(utterance)
        elif intent == 'Intent.FrozenLoadingIssue':
            Utterances.frozenloading.append(utterance)
        elif intent == 'Intent.GRMIssues':
            Utterances.grm.append(utterance)
        elif intent == 'Intent.GRSIssues':
            Utterances.grs.append(utterance)
        elif intent == 'Intent.MobileManagement':
            Utterances.mobilemanagement.append(utterance)
        elif intent == 'Intent.NetworkIssues':
            Utterances.network.append(utterance)
        elif intent == 'Intent.OutlookIssues':
            Utterances.outlook.append(utterance)
        elif intent == 'Intent.RatingIssues':
            Utterances.rating.append(utterance)
        elif intent == 'Intent.HardWareIssues':
            Utterances.hardware.append(utterance)
        elif intent == 'None':
            Utterances.none.append(utterance)
        else:
            print(f'Something is wrong with training.json. Mislabeled intent. {intent}')
    f.close() 
    return Utterances

    
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
