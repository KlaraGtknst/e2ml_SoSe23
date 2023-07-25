# Module for auxiliary functions.

import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display, Markdown, Latex

def display_dataframe(table):
    n = len(table)
    header = [x for x in table.columns][1:]
    
    out = '| ' + ' | '.join([str(y) for y in header]) + " |\n"
    out += '| --- '*len(header) + "|\n"
    
    for x in table.iterrows():
        ys = list(x[1])[1:]
        out += '| ' + ' | '.join([str(y) for y in ys]) + " |\n"
    
    display(Markdown(out))
    

def displayTable(table, header=None, side=None, caption=""):
    n = len(table)
    out = '| --- '*len(table[0]) + '|'
    
    if header != None:
        out = ''.join(['| ', ' |'.join(header), ' |', "\n"] ) + out
    else:
        out = '| '*len(table[0]) + "|\n" + out
    
    if side != None:
        out =  '|  ' + out
        out += " --- |\n"
        for i in range(n):
            out += ''.join(['| **', side[i] ,'** | ', ' | '.join([str(z) for z in table[i]]), ' |', "\n"])
    else:
        out += "\n"
        for i in range(n):
            out += ''.join([' | '.join([str(z) for z in table[i]]), ' |', "\n"])
    
    if caption!=None:
        out = '#### ' + caption + "\n" + out
    
    display(Markdown(out))
    
def displayFullEval(cm):
    bare = [("Accuracy", cm.acc()), ("Error rate", cm.errorRate()), ("Scott's $\pi$", cm.scottPi()), ("Cohen's $\kappa$", cm.cohenKappa())]
    
    names = ["True-Positive rate", "False-Negative rate", "Likelihood ratio, positive", "Likelihood ratio, negative", "$f_1$-measure"]
    tprs = [cm.tpr(i) for i in range(cm.n)]
    fnrs = [cm.fnr(i) for i in range(cm.n)]
    sens = [cm.lrP(i) for i in range(cm.n)]
    spec = [cm.lrN(i) for i in range(cm.n)]
    fOne = [cm.fMeasure1(i) for i in range(cm.n)]
    
    cm.display()
    displayTable([[x[1] for x in bare]], header=[x[0] for x in bare])
    displayTable([tprs, fnrs, sens, spec, fOne], header=[x for x in cm.targets], side=names)
