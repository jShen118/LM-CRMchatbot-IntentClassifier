'''
This is a python script to read data from training.json to split utterances by their intents into separate lists.
There's also a function to write to labeled utterances to training.json
There should be 411-ish total utterances that span over 11 intents (including None):
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
def readtrainingjson():
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

#data is list<(<utterance>, <intent>)>
def writetrainingjson(data):
    # Read JSON file
    with open('training.json') as trainingjson:
        jsonData = json.load(trainingjson)
    for d in data:
        jsonObject = {
            'text': d[0],
            'intentName': d[1],
            'entityLabels': []
        }
        jsonData.append(jsonObject)
    with open('training.json','w') as trainingjson :
        json.dump(jsonData, trainingjson, indent=4, separators=(',',': '))
    
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
