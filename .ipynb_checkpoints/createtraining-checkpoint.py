import pandas
import IntentClassifier as ic
import json
import random
#pandas.read_excel(r'C:\Users\n1555085\Downloads\Copy of CSAT Help hub April Responses.xlsx')

'''
path has to be an excel sheet exported from power bi 
https://app.powerbi.com/groups/me/reports/58c36995-df6c-4d31-a5e2-ebd4c77c2290/ReportSection60735aa100a6ccfbebee
'''
def createTraining(xlpath, numPerIntent=10):
    sheet = pandas.read_excel(xlpath)
    utterances = sheet['Unnamed: 5'].dropna().values.tolist()[1:]
    labels = ic.classifyMultiple(ic.SVMpredictMultiple(utterances))
    allData = list(zip(utterances, labels))
    random.shuffle(allData)
    intentsDict = {
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
    data = []
    for d in allData:
        if len(data) == 11*numPerIntent:
            break
        intentLabel = d[1][0]
        if intentsDict[intentLabel] < numPerIntent:
            data.append(d)
            intentsDict[intentLabel] += 1
            
    def sortByIntent(data):
        access = [d for d in data if d[1][0] == 'Intent.AccessIssues']
        callquality = [d for d in data if d[1][0] == 'Intent.CallQualityIssues']
        frozenloading = [d for d in data if d[1][0] == 'Intent.FrozenLoadingIssue']
        grm = [d for d in data if d[1][0] == 'Intent.GRMIssues']
        grs = [d for d in data if d[1][0] == 'Intent.GRSIssues']
        mobilemanagement = [d for d in data if d[1][0] == 'Intent.MobileManagement']
        network = [d for d in data if d[1][0] == 'Intent.NetworkIssues']
        outlook = [d for d in data if d[1][0] == 'Intent.OutlookIssues']
        rating = [d for d in data if d[1][0] == 'Intent.RatingIssues']
        hardware = [d for d in data if d[1][0] == 'Intent.HardWareIssues']
        none = [d for d in data if d[1][0] == 'None']
        return access+callquality+frozenloading+grm+grs+mobilemanagement+network+outlook+rating+hardware+none
        
    data = sortByIntent(data)
    jsonData = []
    for d in data:
        #d[0] is utterance
        #d[1][0] is intent label
        #d[1][1] is prediction score of intent label
        jsonObject = {
            'text': d[0],
            'intentName': d[1][0],
            'entityLabels': []
        }
        jsonData.append(jsonObject)
    with open('newTraining.json','w') as newjson :
        json.dump(jsonData, newjson, indent=4, separators=(',',': '))









