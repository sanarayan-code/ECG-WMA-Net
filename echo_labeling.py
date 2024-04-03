# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

# method to extract left ventricle section of text
def get_LV(x):
    x = str(x)
    if ". left ventricle " in x:
        x = x.split(". left ventricle ", 1)[1]

        stop_phrases = ["right ventricle", "atria", "tricuspid valve", "pericardium", "mmode"]

        for phrase in stop_phrases:
            x = x.split(phrase, 1)[0]

        return x.lower()
    else:
        return None

def or_compile(term_list):
    ret = "(" + term_list[0]
    for i in range(1, len(term_list)):
        ret += "|"
        ret += term_list[i]
    ret += ")"
    return re.compile(ret)

#class for easier handling of terms in the term map
class TermClass:
    def __init__(self, expression, column):
        self.expr = expression
        self.col = column

# create map of terms and regex using provided phrases
terms = {}
ef55_terms = ["mildly reduced", "mildly.moderately reduced", "4[6-9]\s?%", "5[0-5]\s?%"]
ef45_terms = ["moderately reduced", "moderately.markedly reduced", "4[0-4]\s?%", "3[6-9]\s?%", "35\s?-\s?45%"]
ef35_terms = ["markedly reduced", "markedly.severely reduced", "3[0-4]\s?%", "2[6-9]\s?%", "25\s?-\s?35%"]
ef25_terms = ["severely reduced", "2[0-4]\s?%", "[1\s][0-9]\s?%", "<\s?25\s?%"]
global_terms = ["(global|diffuse) hypokinesis"]
apex_terms = ["apical"]
posterior_terms = ["posterior", "postero"]
inferior_terms = ["inferior", "infero"]
septum_terms = ["septal"]
lateral_terms = ["lateral"]
anterior_terms = ["anterior", "antero"]
abnormal_terms = ["(?<!no )regional wall.?motion abnormalities"]
normal_terms = ["systolic function (is |are )?(grossly )?normal",
                "contractility (is |are )?normal",
                "normal (in )?structure and function",
                "wall.?motion (is )?normal",
                "global (left ventricular |lv )?function (is )?normal",
                "no (segmental )?wall motion (abnormality|abnormalities)"]


terms = {"kinesis":     [TermClass(or_compile(global_terms), "global"),
                                TermClass(or_compile(apex_terms), "apex"),
                                TermClass(or_compile(posterior_terms), "posterior"),
                                TermClass(or_compile(inferior_terms), "inferior"),
                                TermClass(or_compile(septum_terms), "septum"),
                                TermClass(or_compile(lateral_terms), "lateral"),
                                TermClass(or_compile(anterior_terms), "anterior")],
         "aneurysm":    [TermClass(or_compile(apex_terms), "apex"),
                                TermClass(or_compile(posterior_terms), "posterior"),
                                TermClass(or_compile(inferior_terms), "inferior"),
                                TermClass(or_compile(lateral_terms), "lateral")],
         "abnormal":    [TermClass(or_compile(abnormal_terms), "abnormal")]}

def loop_apply(lv_text, terms_dict):
    # do the above but loop and create separate pd for more efficient memory usage
    vals = {}
    for phrase in terms_dict.keys():
        for keyword in terms_dict[phrase]:
            vals[keyword.col] = []

    # cols - keep track of cols to set to 1 for this sentence\
    cols = set()

    for sentence in lv_text:
        if sentence is not None:
            #split into parts
            splits = sentence.split(". ")
            for s in splits:
                #next iterate through the phrases
                for phrase in terms_dict.keys():
                    if phrase in s:
                        for keyword in terms_dict[phrase]:
                            #regex match to determine if should be 1
                            if len(keyword.expr.findall(s))>0:
                                cols.add(keyword.col)

        #special conditions
        # ef_set = set(["ef_25", "ef_35", "ef_45", "ef_55"])
        # if not cols.isdisjoint(ef_set):
        #     cols.add("global")

        abnormal_set = set(["apex", "posterior", "inferior", "septum", "lateral", "anterior"])
        if not cols.isdisjoint(abnormal_set):
            cols.add("abnormal")

        #parse cols filled in above to make rows with 1s and 0s
        for header in vals.keys():
            if header in cols:
                vals[header].append(1)
            else:
                vals[header].append(0)

        cols.clear()

    return pd.DataFrame(vals)

def runAllFromCSV(filename):
    df = pd.read_csv(filename)

    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')

    df["LV"] = df["TEXT"].map(get_LV)

    df_result = df.join(loop_apply(np.array(df["LV"]), terms))

    writename = filename.split(".csv")[0] + "_labeled.csv"

    # writes result df to csv, comment below line if not needed
    df_result.to_csv(writename)

    return df_result

# could read in created csv or just set to method output
df_labeled = runAllFromCSV("/content/echo_note_sample.csv")  #create labels from an echo note sample. 
