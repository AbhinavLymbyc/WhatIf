################################################
########### KDA - What if Analysis #############
################################################

#packages
import pandas as pd
import numpy as np
import os
import json
import math
import sys

#Reading input to the code
work_dir = sys.argv[1]

#setting working directory
os.chdir(work_dir)

#loading input
inp = json.load(open('input.json'))

def bestinclass (inp):
    data1 = pd.DataFrame()
    model_data = pd.DataFrame(inp["kda"]["model"])
    for i in range(0,len(inp["best_in_class"])):
        s = str(inp["best_in_class"][i]["label"])
        d1 = pd.DataFrame(inp["best_in_class"][i]["data"])
        beta = float(model_data['beta_coef'][model_data['code']==s])
        if(beta > 0):
            o = math.ceil(max(d1["final_wgt"]))
        elif(beta < 0):
            o = math.ceil(min(d1["final_wgt"]))
        d1 = pd.DataFrame([s,o],index=['label','optimal']).transpose()
        data1 = pd.concat([data1,d1])
    return data1 


def whatifanalysis (inp,bic,bic_inp=0):

    ## Step 0: Creating inputs for the function
    
    if(inp["type"] == "statement"):
        var = pd.DataFrame(inp["variables"])[["label","change","actual"]]
    elif(inp["type"] == "theme"):
        var1 = pd.DataFrame(inp["variables"])
        var = pd.DataFrame()
        for i in range(0,len(inp["summarized_target_values"])):
            desc = str(inp["summarized_target_values"][i]["desc"])
            val = int(var1[var1["desc"] == desc]["change"])
            act = int(var1[var1["desc"] == desc]["actual"])
            var2 = pd.DataFrame(inp["summarized_target_values"][i]["values"])
            var2["change"] = val
            var2["actual"] = act
            var = pd.concat([var,var2])
        del var1, var2, val, desc
        var = var[["label","change","actual"]]

    if(bic==1):
        var = pd.merge(var, bic_inp, how = 'outer', on = 'label')
        var["change"] = var["optimal"] - var["actual"]
        bic_output = var[["label","optimal"]]
        
    var = var[["label","change"]]
    brand = str(inp['target'])
    data1 = pd.DataFrame(inp["input"])
    model_data = pd.DataFrame(inp["kda"]["model"])
    for i in inp['eql']['dimension'].keys():
        if len(inp['eql']['dimension'][i]) != 1:
            if inp['eql']['dimension'][i]['variable_type'] == "dependent":
               dep_var = str(i)
               break

    ## Step 1: Converting input Statements data into usable form

    data2a = data1[list((set(data1.columns) & set(list(model_data.code))))]
    names = list(data2a.columns)
    data2a = data2a.fillna(0)

    def index_containing_substring(the_list, substring):
        indices = [i for i, s in enumerate(the_list) if substring.lower() in s.lower()]
        if not indices:
            return 0
        return 1

    d1=pd.DataFrame()

    for i in names:
        data3a = data2a[data2a[i] == 0]
        data3a = data3a[[i]]
        data3b = data2a[data2a[i] != 0]
        data3b = data3b[[i]]
        data3 = data3b[i].apply(lambda y: index_containing_substring(y,brand) )
        data4 = data3a.append(pd.DataFrame(data = data3), ignore_index = True)
        d1 = pd.concat([d1,data4],axis=1)

    del names, data2a, data3, data3a, data3b, data4, i
    
    ## Step 2: Getting Distribution of Dependent variable from actual data to decide the threshold for Predicted Probability

    data2b = data1[dep_var].fillna("no_brand")
    data2b = data2b.apply(lambda y: 1 if(brand.lower() in y.lower()) else 0)
    #dist_act = data2b.sum() * 100 / len(data2b)
    dist_act = data2b.sum()

    del data2b

    ## Step 3: Calculating Predicted Probability of Dependent variable using the model equation

    md1 = model_data[model_data["Variables"] != "intercept"]
    md1.index = md1["code"]
    md1 = md1["beta_coef"]
    d2 = d1[md1.index]

    o1 = np.array(np.dot(d2,md1), dtype=np.float32)

    d2["log_odds"] = o1 + model_data["beta_coef"][model_data["Variables"] == "intercept"].values
    d2["odds"] = np.exp(d2['log_odds'])
    d2["pred_prob"] = d2['odds'] / (1 + d2['odds'])

    del md1, o1

    ## Step 4: Sorting the data by predicted probability of dependent variable and finding the threshold using original distrisbution of dependent variable

    d3 = d2.sort_values(["pred_prob"], ascending = [0]).reset_index(drop=True)
    thres = d3['pred_prob'][dist_act]
    d3["pred_dep"] = np.where(d3["pred_prob"] > thres,1,0)

    ## Step 5: Creating updated dataset with imputing values for the statements based on the requested user change

    var1 = var[var['label'].isin((set(list(var['label'].values)) & set(list(model_data.code))))]
    a1 = d3[d3['pred_dep'] == 0]
    a2 = d3[d3['pred_dep'] == 1]

    for i in var['label']:
        if i in list(model_data.code):
            change = float(var[var['label'] == i]['change'])
            beta = float(model_data[model_data['code'] == i]['beta_coef'])
            chg_req = int(len(d3) * abs(change/100))
            
            if change * beta > 0:
                a1 = a1.sort_values(["pred_prob"], ascending = [0])
                b1 = a1[a1[i] == 0]
                b2 = a1[a1[i] == 1]
                if ((change > 0) & (len(b1) != 0)):
                    chg_req = int(np.where(len(b1) < chg_req,len(b1),chg_req))
                    b1a = b1[:chg_req]
                    b1b = b1[chg_req:]
                    b1a[i] = 1
                    b1 = pd.concat([b1a,b1b])
                elif ((change < 0) & (len(b2) != 0)):
                    chg_req = int(np.where(len(b2) < chg_req,len(b2),chg_req))
                    b2a = b2[:chg_req]
                    b2b = b2[chg_req:]
                    b2a[i] = 0
                    b2 = pd.concat([b2a,b2b])
                a1 = pd.concat([b1,b2]).sort_values(["pred_prob"], ascending = [0])
            elif change * beta < 0:
               a2 = a2.sort_values(["pred_prob"], ascending = True)
               b1 = a2[a2[i] == 0]
               b2 = a2[a2[i] == 1]
               if ((change > 0) & (len(b1) != 0)):
                    chg_req = int(np.where(len(b1) < chg_req,len(b1),chg_req))
                    b1a = b1[:chg_req]
                    b1b = b1[chg_req:]
                    b1a[i] = 1
                    b1 = pd.concat([b1a,b1b])
               elif ((change < 0) & (len(b2) != 0)):
                    chg_req = int(np.where(len(b2) < chg_req,len(b2),chg_req))
                    b2a = b2[:chg_req]
                    b2b = b2[chg_req:]
                    b2a[i] = 0
                    b2 = pd.concat([b2a,b2b])
               a2 = pd.concat([b1,b2]).sort_values(["pred_prob"], ascending = [0])

    d4 = pd.concat([a1,a2])        
        
    ## Step 6: Recalculating new predicted probability for the dependent variable for the new dataset

    md1 = model_data[model_data["Variables"] != "intercept"]
    md1.index = md1["code"]
    md1 = md1["beta_coef"]
    d4a = d4[md1.index]
    
    o1 = np.array(np.dot(d4a,md1), dtype=np.float32)

    d4["log_odds_new"] = o1 + model_data["beta_coef"][model_data["Variables"] == "intercept"].values
    d4["odds_new"] = np.exp(d4['log_odds_new'])
    d4["pred_prob_new"] = d4['odds_new'] / (1 + d4['odds_new'])
    d4 = d4.sort_values(["pred_prob_new"], ascending = [0]).reset_index(drop=True)
    d4["pred_prob_new"][:dist_act] = d4["pred_prob_new"][:dist_act] + 0.000001
    d4["pred_dep_new"] = np.where(d4["pred_prob_new"] > thres,1,0)
    
    del md1, o1, d4a
        
    ## Step 7: Distribution of New predicted dependent
    output = math.ceil((float(d4["pred_dep_new"].sum()) * 100) / float(len(d4)))
    
    #For What if
    if(bic==0):
        return output
    #For Best in Class
    elif(bic==1):
        result = [output,bic_output]
        return result

bic_input = bestinclass(inp)
wia_output = whatifanalysis(inp=inp,bic=0)
bic_output = whatifanalysis(inp=inp,bic=1,bic_inp=bic_input)
bic_value = bic_output[0]
bic = pd.DataFrame.to_json(bic_output[1], orient = "records")

result = json.dumps({'what_if':wia_output, 'best_in_class_value':bic_value, 'best_in_class':bic},ensure_ascii=False)
print (result)
############################ End of Code ############################