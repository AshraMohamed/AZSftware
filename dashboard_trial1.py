# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:23:33 2022

@author: OlgaShapran
"""


from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
#st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np

#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
#from sklearn.metrics import precision_score, recall_score
#import pickle
#import AZ_utils
#import seaborn as sns
#import os
#from sklearn import tree	
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split

#from matplotlib import pyplot as plt
#from sklearn.metrics import plot_confusion_matrix
#from sklearn import metrics
#from sklearn.ensemble import GradientBoostingClassifier

import base64
from PIL import Image
from bokeh.sampledata.autompg import autompg_clean as df
import panel as pn
pn.extension('tabulator')
PALLETTE=["#ff6f69","ffcc5c","#88d8b0"]
import scipy.sparse as sp
import math
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
pd.set_option("display.max_columns", 32)


def main():
    return "<p>trial here!</p>"

#country=['Total Universe','Reached']
#population=[6000000, 4500000]
#x=country
#y=population
#variance=[2000000,3000000]
#colors = ['grey', 'red']
#
## function to add value labels
#def addlabels(x,y):
#    for i in range(len(x)):
#        plt.text(i,y[i],y[i])
#plt.style.use('dark_background')
#plt.bar(country, population, width=0.6, color=colors, yerr=variance)
# # calling the function to add value labels
#addlabels(country, population)
#
#  # giving title to the plot
#plt.title("Exposed Users")
#      
#    # giving X and Y labels
##plt.xlabel("Courses")
#plt.ylabel("'000' population")
#plt.show()

