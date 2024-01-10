# -*- coding: utf-8 -*-
"""
@author: OLGA_TYAN_SHANSS @ 2022

#
"""
from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

from sklearn.metrics import precision_score, recall_score
import pickle
import AZ_utils
import seaborn as sns
import os
from sklearn import tree	
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import base64
from PIL import Image



  

def main():
      
    
    @st.cache_resource(experimental_allow_widgets=True) #allow_output_mutation=True
    # @st.cache(allow_output_mutation=True) #allow_output_mutation=True
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    def set_png_as_page_bg(png_file):
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = '''
        <style>
        .stApp {
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
        }
        </style>
        ''' % bin_str
        
        st.markdown(page_bg_img, unsafe_allow_html=True)
        return
    #st.set_page_config(layout="wide")
    
    @st.cache_data(persist="disk")  #persist= True
    # @st.cache(persist= True)  #persist= True
    def load():
        
        if uploaded_file is not None:
            d=pd.read_excel(uploaded_file)
        else:
            #pic = "C:/Users/blublazer70/Desktop/Olga/d_raw.dat"
            #pic="d_raw.dat"
            pic= "Final_TV_Panel_All_03.05.pkl"
           
            with open(pic, "rb") as f:
                d=pickle.load(f)    
        return d
   
    @st.cache_resource(experimental_allow_widgets=True)     #allow_output_mutation=True
    # @st.cache(allow_output_mutation=True)    #
    def convert_df(X,Y):
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:  
            X.to_excel(writer, sheet_name='X')
            Y.to_excel(writer, sheet_name='Y')
            writer.close()
            return buffer
            
    def plot_metrics(metrics_list,lab):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            from yellowbrick.classifier import confusion_matrix
            confusion_matrix(clf, X_train, y_train, X_test, y_test,percent=False,classes=lab)
            plt.tight_layout()
            st.pyplot()
       
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curves")
            from yellowbrick.classifier import ROCAUC
            viz = ROCAUC(clf,fig=plt.figure(figsize=(7,5)),classes=lab)
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.show();
            st.pyplot()
            
        if "Classification report" in metrics_list:
            
            st.subheader("Classification Report")
            from yellowbrick.classifier.classification_report import classification_report
            classification_report(clf,X_train, y_train,X_test, y_test,
                                  support="percent",
                                  cmap="Reds",
                                  font_size=16,
                                  fig=plt.figure(figsize=(8,6)),
                                 classes=lab);
            st.pyplot()   
       
        if "Class prediction error" in metrics_list:
            st.subheader("Class Prediction Error")
            from yellowbrick.classifier import ClassPredictionError
            visual=ClassPredictionError(clf,classes=lab)
            
            visual.fit(X_train, y_train)
            visual.score(X_test,y_test)
            visual.show();
            st.pyplot()
   
    def plot_importances(importances_list,lab):
        if "Impurity Decrease" in importances_list:
            st.subheader("Feature Importances-Impurity Decrease")
            feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
            feat_importances.nlargest(20).sort_values(ascending=True).plot(kind='barh')
            st.pyplot()
        
        if "Permutation Importance" in importances_list:
            st.subheader("Feature Importances-Permutation")
            from sklearn.inspection import permutation_importance
            #use only on test data
            perm_importances = permutation_importance(clf, X_test, y_test)
            perm_importances= pd.Series(perm_importances.importances_mean,index=X.columns)
            perm_importances.nlargest(20).sort_values(ascending=True).plot(kind='barh')
            st.pyplot()
         
        if "SHAP Summary" in importances_list:    
            import shap
            st.subheader("Feature Importances-Shapley Additive")
            #for all classes stacked 
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test,class_names=lab)
            st.pyplot() 
           
    def plot_importances_L(importances_list,lab):
        if "SHAP Tree" in importances_list:
            import shap
            n_class = st.slider('Choose class', 0, y_bins-1,step=1,key=1)
            st.subheader("SHAP Tree for Class "+lab[n_class])
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values[n_class], X_test)
            st.pyplot()
        
        if "SHAP Waterfall" in importances_list:
            import shap
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            n_class_ = st.slider('Choose class', 0, y_bins-1,step=1,key=2)
            n_element = st.number_input('Insert a household number up to '+str(len(X)),min_value=0, max_value=len(X), value=100, step=1)
            st.subheader("SHAP Waterfall for Class "+lab[n_class_]+" Household:"+str(n_element))
            
            shap.waterfall_plot(shap.Explanation(values=shap_values[n_class_][n_element], 
                                                  base_values=explainer.expected_value[n_class_], data=X_test.iloc[n_element],  
                                             feature_names=X_test.columns.tolist()))
            st.pyplot()
           
    root= AZ_utils.__file__
    root=os.path.dirname(root)
    st.write(root)
    os.chdir(root)
    
    set_png_as_page_bg('datascience-1024x396-1.png')
    
    st.title("AZ TAM – Control Parameters and Panel Distribution Software")
    st.title(" Multi-Class Classification @2022")

    with st.sidebar.container():
        image = Image.open('Original.png')
        st.sidebar.image(image, use_column_width=True)
    
    st.sidebar.title("I.Data Loading & Feature Engineering")
    #st.sidebar.markdown("")
    
    uploaded_file = st.sidebar.file_uploader(type=["xlsx"],label="Load raw data else default data will be used.Data should be in the same format as 'Final_TV_Panel_All_03.05.csv'") 
    
    st.write('----------PROGRESS-------------')
    d=load()
    
    st.write('DATA LOADING-->COMPLETED')
        
    y_to_use = st.sidebar.selectbox('Choose Dependent Variable[Target]-Note:If provided should be the last column in loaded dataset',('TS_All_view','TS_weekdays_view','TS_weekends_view','MARSA (AZ)_all_view','TS_Daypart_06_12_Binary','TS_Daypart_12_18_Binary','TS_Daypart_18_00_Binary','TS_Daypart_00_06_Binary'))
    
    y_to_disc=['TS_All_view','TS_weekdays_view','TS_weekends_view','MARSA (AZ)_all_view']
    
    
    if y_to_use in y_to_disc :
        
        no_bins = st.sidebar.selectbox('No Bins-Target quantile binning',( '3', '5','2','7'))
    
        y_bins=int(no_bins)
        
        if y_bins==2:
                lab=["LV","HV"]
        elif y_bins==3:
                lab=["LV","AV","HV"]
        elif y_bins==5:
                lab=["LLV","LV","AV","HV","HHV"]
        elif y_bins==7:
                lab=["LLLV","LLV","LV","AV","HV","HHV","HHHV"]       
    else: 
        y_bins=2
        lab=["NOT WATCH","WATCH"]
    
    #st.write(y_bins)
    #y_bins=3
    target_binning='frequency'  #target_binning=='equal_width' or 'frequency'
    
    Y, X =AZ_utils.data_preprocessing(d,y_bins,target_binning)
    st.write('DATA PREPROCESSING-->COMPLETED')
    #st.write('DATA PREPROCESSING_AFTER')
    
    total_hh=len(X)
    zero_tv_hh=len(X[X['Number_of_Tv_sets']==0])
    
    #buffer = convert_df(X,Y)
    
    
    st.sidebar.download_button(
            
            label="Download processed data",
            #data=buffer,
            data=convert_df(X,Y),
            file_name="pandas_multiple.xlsx",
            mime="application/vnd.ms-excel" )
    #st.write('BEFORE SUMMARY STATS')
    sex_Age_matrix=AZ_utils.summary_stats(d)
    #st.write('AFTER SUMMARY STATS')
    sex_Age_matrix_perc=100*(sex_Age_matrix/sex_Age_matrix[-1])
    
    sex_matrix=X[['Number_of_males','Number_of_females']].sum()  
    
    tv_sets=X['Number_of_Tv_sets'].value_counts(bins =  [-1,0,2,10]).sort_index()
    
    tv_sets=tv_sets.rename({tv_sets.index[0]: "0",tv_sets.index[1]: "1 to 2",tv_sets.index[2]: " More than 2"}, axis='index')
     
    hh_siz=X['Number_of_HH_Members'].value_counts(bins = [0,1,2,20]).sort_index()
    
    hh_siz=hh_siz.rename({hh_siz.index[0]: "1 Member HH",hh_siz.index[1]: " 2 Member HH",hh_siz.index[2]: "Larger than 2 Member HH"}, axis='index')
    
    main_smart=pd.to_numeric(d['B5_1'],errors='coerce').fillna(4).value_counts()
    
    internet=d['B29'].value_counts()

    if y_to_use=='MARSA (AZ)_all_view':
        col='MARSA Provided [PerWeekPerMember]'
        y=Y[col]
        y=y.astype('int64')
        
    elif y_to_use=='TS_All_view':
        col='All_Tyan_ShanSS_provided[PerWeekPerMember_normalised]'
        y=Y[col]
        y=y.astype('int64')
    
    elif y_to_use=='TS_weekdays_view':
        col='Weekdays_Tyan_ShanSS_provided[PerWeekPerMember_normalised]'
        y=Y[col]
        y=y.astype('int64')
        
    elif y_to_use=='TS_weekends_view':
        col='WeekEnds_Tyan_ShanSS_provided[PerWeekPerMember_normalised]'
        y=Y[col]
        y=y.astype('int64')
    ####
    else:  
        y=Y[y_to_use]
        y=y.astype('int64')
   
    
    feat_select = st.sidebar.selectbox("Choose Independent Variables[Features]", ("Default","Custom","All","All_Exclude_CH", "All_Exclude_Intro","All_Exclude_Intro_CH"))

    if "All_Exclude_CH" in feat_select:
        col_start="CH_1"
        col_end="CH_50"
        idx_start=X.columns.get_loc(col_start)
        idx_end=X.columns.get_loc(col_end)
        X.drop(X.iloc[:,idx_start:idx_end+1],inplace = True, axis = 1)
    
    if "All_Exclude_Intro" in feat_select:
        col_start="Intro_A1_1"
        col_end="Intro_A1_8"
        idx_start=X.columns.get_loc(col_start)
        idx_end=X.columns.get_loc(col_end)
        X.drop(X.iloc[:,idx_start:idx_end+1],inplace = True, axis = 1)
    
    if "All_Exclude_Intro_CH" in feat_select:
        col_start="CH_1"
        col_end="CH_50"
        idx_start=X.columns.get_loc(col_start)
        idx_end=X.columns.get_loc(col_end)
        X.drop(X.iloc[:,idx_start:idx_end+1],inplace = True, axis = 1)
        col_start="Intro_A1_1"
        col_end="Intro_A1_8"
        idx_start=X.columns.get_loc(col_start)
        idx_end=X.columns.get_loc(col_end)
        X.drop(X.iloc[:,idx_start:idx_end+1],inplace = True, axis = 1)
        
        #'MaritalStatus_num_single', 'MaritalStatus_num_divorced', 'MaritalStatus_num_married', 'MaritalStatus_num_widowed', ,'EmploymentStatus_num_working','Employment_num_unemployed','Employment_num_housewife', ''totalNum_Terrestrial', 'totalNum_Satellite', 'totalNum_Cable/Digital'','Employment_num_retired', 'Employment_num_student','MaritalStatus_num_single', 'MaritalStatus_num_divorced', 'MaritalStatus_num_married', 'MaritalStatus_num_widowed',
    if "Default" in feat_select:
        X_labels=['Number_of_Tv_sets','Total_SmartTVs', 'Number_of_HH_Members', 'InternetAccess_via_connection','SES', 'Head_of_Family_position', 'Financial_Status', 'isSecondHouse', 'Current_address_Duration','Employment_status_HeadofHH','Marital_status_HeadofHH','Household_reception_type','Presence_of_Children(0-16)']
        X=X[X_labels] 
         
    if "Custom" in feat_select:  
        feat_selection = st.sidebar.multiselect("Choose at LEAST 5 features to continue", (list(X.columns))) 
        #feat_selection = list(set(feat_selection)) #remove duplicates
        #if len(feat_select)>=5:
        X=X[feat_selection]
            
    if st.sidebar.checkbox("Display input data snapshot ", False):
        st.subheader("Clean Data/Feature Engineering")
        #st.write(X.head(2))
        
        #st.table(X.dtypes.astype(str))
        st.dataframe(X.head()) 
   
    if st.sidebar.checkbox("Display selected features", False):
        st.subheader("Selected Independent Variables")
        #st.write(X.head(2))
        
        #st.table(X.dtypes.astype(str))
        st.table(X.columns)
    
    if  not "Custom" in feat_select or len(feat_selection)>=5:
        st.sidebar.write("Ensemble Algorithm:Random Forest")
        testperc= st.sidebar.slider('Test_set(%)', 0.1,0.5,step=0.1,value=0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=testperc, random_state=0,shuffle=True)
        
                
        algo='Random Forest'
        
        
        @st.cache_resource(experimental_allow_widgets=True) 
        # @st.cache_resource()  ##allow_output_mutation=True,suppress_st_warning=True
        # @st.cache(allow_output_mutation=True,suppress_st_warning=True)  ##allow_output_mutation=True,suppress_st_warning=True
        def train_model(X_train,y_train,algo):
            if algo=='Random Forest':   
                clf = RandomForestClassifier(class_weight='balanced_subsample',random_state=4,n_estimators=200,criterion='gini',max_leaf_nodes=15,min_samples_leaf=10)
            else:
                from sklearn.multiclass import OneVsRestClassifier
                clf =  OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0))
            clf = clf.fit(X_train, y_train)
            
            return clf 
        
        #st.write('BEFORE_TRAINING')
        clf=train_model(X_train,y_train,algo )
        st.write('MODEL TRAINING-->COMPLETED')
        st.write('-----------------------------')
        
        
        if st.sidebar.checkbox("Display model params", False):
            #st.subheader("Features")
            #st.write(X.head(2))
            st.subheader("Model Parameters")
            #st.table(X.dtypes.astype(str))
            #st.table(pd.DataFrame.from_dict(clf.get_params(),orient='index'))   
            st.write(str(clf.get_params()))   
        
        
        st.sidebar.title("II.Classifier Evaluation")
        
        metrics_ = st.sidebar.multiselect("What metrics to display?", ("Confusion Matrix", "ROC Curve", "Classification report","Class prediction error"))  
        
        plot_metrics(metrics_,lab)
        
        
        st.sidebar.title("III.Feature (Control Variables) Importances")
        
        importance_ = st.sidebar.multiselect("Global", ("Impurity Decrease", "Permutation Importance", "SHAP Summary"))  
        #st.write('PLOT IMPORTANCES_G')
        plot_importances(importance_,lab)
        
        importance_L = st.sidebar.multiselect("Local", ("SHAP Tree", "SHAP Waterfall"))  
        
        plot_importances_L(importance_L,lab)
        
        st.sidebar.title("IV.Panel Distribution Matrices")
        
        if st.sidebar.checkbox("Default Distributions", True):
            
            plt.figure()
            s_s=pd.concat([sex_matrix,100*(sex_matrix/sex_matrix.sum())],axis=1)
            s_s.columns=["Total","% Total"]
            s_s.loc["Total"] = s_s.sum()
            st.subheader("Gender distribution")
            #st.write(sns.heatmap(s_a,cmap="YlGnBu",annot=True, cbar=False,fmt='.1f').figure)
            st.write(sns.heatmap(s_s,cmap="coolwarm",annot=True, cbar=False,fmt='.2f').figure)
            #st_cont1 = st.container()
    
            plt.figure()
            s_a=pd.concat([sex_Age_matrix,sex_Age_matrix_perc],axis=1)
            s_a.columns=["Total","% Total"]
            st.subheader("Age/Gender distribution")
            #st.write(sns.heatmap(s_a,cmap="YlGnBu",annot=True, cbar=False,fmt='.1f').figure)
            st.write(sns.heatmap(s_a,cmap="coolwarm",annot=True, cbar=False,fmt='.2f').figure)
            
            plt.figure()
            stat1=pd.concat([tv_sets,100*(tv_sets/tv_sets.sum())],axis=1).round(2)
            stat1.loc["Total"] = stat1.sum()
            #stat.columns=["Total","% Total","Cumsum Total","% Cumsum Total"]
            stat1.columns=["Total","% Total"]
            st.subheader("No TV Sets")
            st.write(sns.heatmap(stat1,cmap="coolwarm", annot=True, cbar=False,fmt='g').figure)
            
            plt.figure()
            stat2=pd.concat([hh_siz,100*(hh_siz/hh_siz.sum())],axis=1).round(2)
            stat2.loc["Total"] = stat2.sum()
            #stat.columns=["Total","% Total","Cumsum Total","% Cumsum Total"]
            st.subheader("HH Size")
            stat2.columns=["Total","% Total"]
            #st.write(X[i].value_counts().sort_index())
            st.pyplot(sns.heatmap(stat2,cmap="coolwarm", annot=True, cbar=False,fmt='g').figure)
            
            plt.figure()
            stat3=pd.concat([main_smart,100*(main_smart/main_smart.sum())],axis=1).round(2)
            #stat3.rename(index={0: "NO", 1: "YES"})
            stat3=stat3.rename({2: "NO", 1: "YES",3:"Don't know",4:"ZERO TV" }, axis='index')
            #stat3.loc["ZERO TV"] =[tv_sets[0], 100*(tv_sets[0]/main_smart.sum())]
            stat3.loc["Total"] = stat3.sum()
            #stat.columns=["Total","% Total","Cumsum Total","% Cumsum Total"]
            st.subheader("Main Smart")
            stat3.columns=["Total","% Total"]
            #st.write(X[i].value_counts().sort_index())
            st.pyplot(sns.heatmap(stat3,cmap="coolwarm", annot=True, cbar=False,fmt='g').figure)
            
            plt.figure()
            stat4=pd.concat([internet,100*(internet/internet.sum())],axis=1).round(2)
            stat4=stat4.rename({2: "NO", 1: "YES",3: "Don't know"}, axis='index')
            stat4.loc["Total"] = stat4.sum()
            #stat.columns=["Total","% Total","Cumsum Total","% Cumsum Total"]
            st.subheader("Internet Access")
            stat4.columns=["Total","% Total"]
            #stat4.rename(index={0.0: "NO", 1.0: "YES"})
            #st.write(X[i].value_counts().sort_index())
            st.pyplot(sns.heatmap(stat4,cmap="coolwarm", annot=True, cbar=False,fmt='g').figure)
            
            
            if st.checkbox("Show Default Dataframes", False):
                st.table(s_s)
                st.table(s_a)
                st.table(stat1)
                st.write(stat2)
                st.write(stat3)
                st.write(stat4)
                
        st.sidebar.header('Frequencies')
        feat_select = st.sidebar.multiselect("Choose features", (list(X.columns))) 
        #st.subheader("Chosen features")
        #st.dataframe(X[feat_select].head())
        
        if len(feat_select)>0:
            
            for i in feat_select:
                st.subheader(i)
                
                stat=pd.concat([X[i].value_counts().sort_index(),100*(X[i].value_counts(normalize=True).sort_index()),X[i].value_counts().sort_index().cumsum(),100*(X[i].value_counts(normalize=True).sort_index().cumsum())],axis=1).round(2)
                stat.columns=["Total","% Total","Cumsum Total","% Cumsum Total"]
                #st.write(X[i].value_counts().sort_index())
                
                plt.figure()
                st.pyplot(sns.heatmap(stat,cmap="coolwarm", annot=True, cbar=False,fmt='g').figure)
                
                if st.checkbox("Show Dataframe"+i, False):
                
                    #st.dataframe(stat)
                    st.table(stat) 
        
        st.sidebar.header('Conditional Frequencies')  
        
        norm = st.sidebar.selectbox('Normalize',('None', 'OverAll', 'OverRows','OverColumns')) 
            
        if norm=='None':
            norm=False
        elif norm=='OverAll':
            norm = True
        elif norm=='OverRows':
            norm = 'index'    
        elif norm=='OverColumns':
            norm ='columns'     
        
        #default_list=      
        row_select = st.sidebar.multiselect("Choose row features", (list(X.columns)),default=None) 
        #row_select=['X.' + x for x in row_select]
        
        column_select = st.sidebar.multiselect("Choose column features", (list(X.columns)))
        #column_select=['X.' + x for x in column_select]
         
        if (len(row_select)>0) and (len(column_select)>0):
            
            drow=[]
            for i in row_select:
                drow.append(X[i])
            
            dcol=[]
            for i in column_select:
                dcol.append(X[i])  
                          
            st.subheader("Heatmap-"+str(row_select)+str(column_select))
            
            tabul=pd.crosstab(drow,dcol,normalize=norm,rownames=row_select,colnames=column_select,dropna=True).round(2)
            plt.figure()
            st.pyplot(sns.heatmap(tabul,cmap="coolwarm", annot=True, cbar=False,fmt='g').figure)
            
            
            if st.checkbox("Show heatmap Dataframe", False):
                #st.dataframe(tabul)   
                st.table(tabul)
            
if __name__ == '__main__':
    main()