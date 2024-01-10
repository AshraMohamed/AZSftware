# -*- coding: utf-8 -*-
"""
@author: OLGA_TYAN_SHANSS @ 2022
"""

import pandas as pd
import numpy as np
import streamlit as st
pd.set_option('mode.chained_assignment',None)

@st.cache_data(persist= True)###suppress_st_warning=True,
def data_preprocessing(d,bins,target_binning):
    #st.write('DATA PREPROCESSING_INSIDE')
    d = d.apply(pd.to_numeric, errors='coerce') #remove strings
    d[d==99]=np.nan
    #d[d.columns[-1]]=d[d.columns[-1]]
    if target_binning=='equal_width':
        y_a=pd.cut(d['Per_week_per_member'], bins,labels=list(range(bins))) #Quantile-based discretization 
    elif target_binning=='frequency':
        y_a=pd.qcut(d['Per_week_per_member'], bins,labels=list(range(bins))) 
    
    #feature encoding 
    #################################
    idx=d.columns.get_loc("B21_1")
    days=d.iloc[:,idx:idx+14]
    days[days>6]=6
    days=6-days
    days_m=days.mean(axis=1)
    
    idx=d.columns.get_loc("B22_1")
    days=d.iloc[:,idx:idx+14]
    days[days>3]=3
    days=3-days
    days_wkd_m=days.mean(axis=1)
    
    idx=d.columns.get_loc("B23_1")
    hours=d.iloc[:,idx:idx+14]
    hours[hours>8]=0
    hours_m=hours.mean(axis=1)
    
    idx=d.columns.get_loc("B24_1")
    hours=d.iloc[:,idx:idx+14]
    hours[hours>8]=0
    hours_wkd_m=hours.mean(axis=1)
    
    view_workdays=days_m*hours_m
    view_workdays=view_workdays/120 #24*5 Weekdays
    view_workdays=view_workdays.fillna(0)
    
    view_wkd=days_wkd_m*hours_wkd_m
    view_wkd=view_wkd/48 #24*2 Weekends
    view_wkd=view_wkd.fillna(0)
    
    view_all=(view_workdays*120+view_wkd*48)/(120+48)
     
    view_d_days=pd.qcut(view_workdays/d.B17, bins,labels=list(range(bins))) 
    view_d_wkd=pd.qcut(view_wkd/d.B17, bins,labels=list(range(bins)))
    view_d_all=pd.qcut(view_all/d.B17, bins,labels=list(range(bins)))
    
    d_interval_sum=0
    for i in range(1,15):
    #print(i)
        col="B26_"+ str(i) +"_1"
        idx=d.columns.get_loc(col)
        d_interval=d.iloc[:,idx:idx+48] #at b26.49 Don’t watch/doesn’t fit
        d_interval= d_interval.fillna(0)  
        if i>1:
        
            d_interval.columns = d_interval_sum.columns
    
    ###############################    
        d_interval_sum=d_interval_sum+d_interval #2
    
    d_interval_sum_single=d_interval_sum.sum(axis=1)#3
    
    ###############################
    d_interval_sum_disc= d_interval_sum.copy() 
    
    ############################### 
    d_interval_sum_disc[d_interval_sum_disc != 0] = 1 #4
    
    d_interval_sum_disc_single=d_interval_sum_disc.sum(axis=1) #5
    ###############################
    
    #viewing day into 4 parts 6-12, 12-18, 12-24,24-6
    
    part=[0,12,24,36]
    daypart=[]
    
    for i in part:
    
        daypart.append(d_interval_sum_disc.iloc[:,i:i+12].sum(axis=1))
        
    #####################################
    daypart=daypart   
    daypart=pd.DataFrame(daypart).T #6
    daypart.columns=['TS_Daypart_06_12','TS_Daypart_12_18','TS_Daypart_18_00','TS_Daypart_00_06']
    
    daypart_disc=daypart.copy()
    daypart_disc[daypart_disc!=0]=1 #7
    daypart_disc.columns=['TS_Daypart_06_12_Binary','TS_Daypart_12_18_Binary','TS_Daypart_18_00_Binary','TS_Daypart_00_06_Binary']
    #####################################
    
    #Channels
    
    #B28_1-B28_50 + 5 extras with txt
    col="B28_1"
    idx=d.columns.get_loc(col)
    channels=d.iloc[:,idx:idx+50]
    
    Y=pd.concat([daypart,daypart_disc], axis = 1)
    
    Y['All_Tyan_ShanSS_provided[PerWeekPerMember_normalised]']=view_d_all
    Y['Weekdays_Tyan_ShanSS_provided[PerWeekPerMember_normalised]']=view_d_days
    Y['WeekEnds_Tyan_ShanSS_provided[PerWeekPerMember_normalised]']=view_d_wkd
    Y['MARSA Provided [PerWeekPerMember]']=y_a
    
    #################################
    ##INPUTS
    col_start="A1_1"
    col_end="A1_8"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp[temp==98]=3
    temp=temp.add_prefix('Intro_')
    X=temp
    
    d.B1[d.B1==6]=0
    
    X['Number_of_Tv_sets']=d['B1']
    
    X['isOneTV_set']=0
    X.isOneTV_set[d.B1==1]=1
    
    X['isMultipleTV_set']=0
    X.isMultipleTV_set[d.B1>1]=1
    
    X['Main_Tv_Onw_years']=d['B3_1'].fillna(d.B3_1.mean())

    col_start="B3_2"
    col_end="B3_5"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp=temp.fillna(temp.mean())
    X['Other_Tv_Own_years']=temp.mean(axis=1).round().fillna(0)
    
    X['Main_Tv_size']=d['B4_1'].fillna(d.B4_1.mean())
    
    X['is_Main_Smart']=d['B5_1']
    X.is_Main_Smart[X.is_Main_Smart!=1]=0
    
    col_start="B5_1"
    col_end="B5_5"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp[temp!=1]=0
    temp=temp.sum(axis=1)
    X['Total_SmartTVs']=temp

    col_start="B9_1_1"
    col_end="B9_1_5"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    X['is_Main_Terrestrial']=temp.sum(axis=1).fillna(0.5)
    
    col_start="B9_1_6"
    col_end="B9_1_9"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    X['is_Main_Satellite']=temp.sum(axis=1).fillna(0.5)
    
    X['is_Main_Cable']=d["B9_1_11"].fillna(0.5)
    
    X['is_Main_Digital']=d["B9_1_12"].fillna(0.5)
    
    temp_a=pd.concat([ X['is_Main_Terrestrial'], X['is_Main_Satellite'], X['is_Main_Cable']+X['is_Main_Digital']],axis=1)
    temp_a.columns=[1,2,3]
    X['Household_reception_type']=temp_a.idxmax(axis=1)

    
    ###
    for i in range(1,6):
        if i>1:
            tempold_ter=temp_ter
            tempold_sat=temp_sat
            tempold_cable=temp_cable
            tempold_digital=temp_digital
            
        col_start="B9_"+str(i)+"_1"
        col_end="B9_"+str(i)+"_5"
        idx_start=d.columns.get_loc(col_start)
        idx_end=d.columns.get_loc(col_end)
        temp_ter=d.iloc[:,idx_start:idx_end+1].sum(axis=1)
        
        col_start="B9_"+str(i)+"_6"
        col_end="B9_"+str(i)+"_9"
        idx_start=d.columns.get_loc(col_start)
        idx_end=d.columns.get_loc(col_end)
        temp_sat=d.iloc[:,idx_start:idx_end+1].sum(axis=1)
        
        temp_cable=d["B9_"+str(i)+"_11"]
        temp_digital=d["B9_"+str(i)+"_12"]
        
        if i>1:
            temp_ter=pd.concat([tempold_ter,temp_ter],axis=1)
            temp_sat=pd.concat([tempold_sat,temp_sat],axis=1)
            temp_cable=pd.concat([tempold_cable,temp_cable],axis=1)
            temp_digital=pd.concat([tempold_digital,temp_digital],axis=1)
                   
    X['totalNum_Terrestrial']=temp_ter.sum(axis=1) 
    X['totalNum_Satellite']=temp_sat.sum(axis=1)
    X['totalNum_Cable/Digital']=temp_cable.sum(axis=1)+temp_digital.sum(axis=1)
    
    ###
    col_start="B10_1_1"
    col_end="B10_1_17"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    cable_service=['Aile TV','Alfanet','ATV+','Azeurotel','Aztelecom','Baktelecom','BB TV','Birlink','KaTV1','Cityline.az','Connect','Sea TV','Sevenline','Smartonline','Smart tv plus','SNTV','Ultel']
    temp.columns=cable_service
    #temp=temp.add_prefix('cab_dig_')
    temp=temp.fillna(0)
    
    X=pd.concat([X,temp], axis = 1)
    
    col_start="B12_1"
    col_end="B12_2"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp=temp.sum(axis=1)
    temp[temp>1]=1
    X['is_Paid_cab_dig']=temp
    
    col_start="B16_1_1"
    col_end="B16_1_4"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp=temp.sum(axis=1)
    temp[temp>1]=1
    X['is_other_Devices_connected']=temp
    
    X['Number_of_HH_Members']=d.B17
    
    X['MembersHH_upto2']=0
    X.MembersHH_upto2[d.B17<2]=1
    X['MembersHH_2+']=0
    X['MembersHH_2+'][d.B17>=2]=1
    
  ################
    col_start="B18_1"
    col_end="B18_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    #temp=d.iloc[:,idx_start:idx_end+2]
    #temp=temp.iloc[:,0:28:2]
    
    col_start="D2_1"
    col_end="D2_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp2=d.iloc[:,idx_start:idx_end+1]
    temp2.columns=temp.columns
    idx=temp[temp==1]
    temp2=temp2[idx==1].sum(axis=1)
    
    temp=d.B18_1.copy()
    temp[temp!=1]=0
    X['is_HH_interviewed']=temp
    
    #temp2=temp*d.D2_1
    X['Marital_status_HeadofHH']=temp2
    temp2=pd.get_dummies(temp2)
    X['HH_single']=temp2[1]
    X['HH_divorced']=temp2[2]
    X['HH_married']=temp2[3]
    X['HH_widowed']=temp2[4]
    
    ################
    col_start="B18_1"
    col_end="B18_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    #temp=d.iloc[:,idx_start:idx_end+2]
    #temp=temp.iloc[:,0:28:2]
    
    col_start="D3_1"
    col_end="D3_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp2=d.iloc[:,idx_start:idx_end+1]
    temp2.columns=temp.columns
    idx=temp[temp==1]
    temp2=temp2[idx==1].sum(axis=1)
    temp2[temp2==7]=0
    temp2[(temp2!=1) & (temp2!=0)]=2
    X['Employment_status_HeadofHH']=temp2
    
    ################

    
    col_start="B19_1"
    col_end="B19_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp=temp[temp==1]
    X['Number_of_males']=temp.sum(axis=1)
    
    temp=d.iloc[:,idx_start:idx_end+1]
    temp=temp[temp==2]
    X['Number_of_females']=temp.sum(axis=1)/2
    
    col_start="B20_1"
    col_end="B20_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    #temp=d.iloc[:,idx_start:idx_end+2]
    #temp=temp.iloc[:,1:28:2]
    
    temp2=temp[temp<12].sum(axis=1)
    temp2[temp2>0]=1
    X['Presence_of_Children(0-12)']=temp2
    
    temp2=temp[temp<16].sum(axis=1)
    temp2[temp2>0]=1
    X['Presence_of_Children(0-16)']=temp2
    
    temp2=temp[(temp >= 12) & (temp<=18)].sum(axis=1)
    temp2[temp2>0]=1
    X['Presence_of_Teen']=temp2
    
    temp2=temp[(temp >18 ) & (temp<=65)].sum(axis=1)
    temp2[temp2>0]=1
    X['Presence_of_Adult']=temp2
    
    temp2=temp[temp >65].sum(axis=1)
    temp2[temp2>0]=1
    X['Presence_of_Senior(65+)']=temp2
    
    col_start="B27_1"
    col_end="B27_4"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1].fillna(0)
    
    temp.columns = ['Language_AZER', 'Language_TURKISH', 'Language_RUSSIAN', 'Language_ENGLISH']
    X=pd.concat([X,temp], axis = 1)
    
    temp=d.iloc[:,idx_start:idx_end+1]
    X['Total_numLanguages_inHH']=temp.sum(axis=1)
    
    col_start="B28_1"
    col_end="B28_50"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp.columns=temp.columns.str.replace('B28','')
    temp=temp.add_prefix('CH').fillna(0)
    X=pd.concat([X,temp], axis = 1)
    
    temp=d.B29
    X['InternetAccess']=temp[temp<2].fillna(0)
    
    ####
    for i in range(1,15):
        if i>1:
            tempold_345G=temp_345G
            tempold_wireless=temp_wireless
            tempold_adsl=temp_adsl
            tempold_fiber=temp_fiber
            
        col_start="B30_"+str(i)+"_3"
        col_end="B30_"+str(i)+"_6"
        idx_start=d.columns.get_loc(col_start)
        idx_end=d.columns.get_loc(col_end)
        temp_345G=d.iloc[:,idx_start:idx_end+1].sum(axis=1)
        
        temp_wireless=d["B30_"+str(i)+"_7"]
        temp_adsl=d["B30_"+str(i)+"_1"]
        temp_fiber=d["B30_"+str(i)+"_2"]
        
        if i>1:
            temp_345G=pd.concat([tempold_345G,temp_345G],axis=1)
            temp_wireless=pd.concat([tempold_wireless,temp_wireless],axis=1)
            temp_adsl=pd.concat([tempold_adsl,temp_adsl],axis=1)
            temp_fiber=pd.concat([tempold_fiber,temp_fiber],axis=1)
                   
    X['totalNum_345G']=temp_345G.sum(axis=1) 
    X['totalNum_wireless']=temp_wireless.sum(axis=1)
    X['totalNum_adsl']=temp_adsl.sum(axis=1) 
    X['totalNum_fiber']=temp_fiber.sum(axis=1) 
    X['InternetAccess_via_connection']=np.where(((X['totalNum_adsl']>0) | (X['totalNum_fiber']>0)),1,0)
    X['InternetAccess_via_connection']=np.where(((X['InternetAccess_via_connection']==0) & ((X['totalNum_345G']>0) | (X['totalNum_wireless']>0)) ),0, X['InternetAccess_via_connection'])
    ####
    
    col_start="B31_1"
    col_end="B31_5_TEXT"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp=temp.iloc[:,1:16:2].fillna(0)
    temp.columns = ['Number_Desktops', 'Number_Laptops', 'Number_Tablets','Number_Smartphone', 'Number_Gamebox']
    X=pd.concat([X,temp], axis = 1)
    
    temp=d.iloc[:,idx_start:idx_end+1]
    temp=temp.iloc[:,1:16:2].fillna(0)
    X['Total_number_Devices_usedforTVViewing']=temp.sum(axis=1)
    
    col_start="B33_1"
    col_end="B33_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    
    X['Number_online_Daily']=temp[temp==1].sum(axis=1).fillna(0)
    X['Number_online_atLeast_Weekly']=temp[temp==2].sum(axis=1).fillna(0)/2
    X['Number_online_atLeast_Monthly']=temp[temp==3].sum(axis=1).fillna(0)/3
    X['Number_online_lessthan_Monthly']=temp[temp==4].sum(axis=1).fillna(0)/4
    
    col_start="B34_1"
    col_end="B34_7"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    online_service=['Netflix','Megogo','İVİ','Okko','HBO','Exxen','Blu TV']
    temp.columns=online_service
    X=pd.concat([X,temp], axis = 1)
    
    col_start="B37"
    col_end="B37"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp[temp<=4]=1
    X['Recording_Freq']=temp.fillna(0)
    
    col_start="D1_1"
    col_end="D1_13"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    
    X['Education_score']=temp.sum(axis=1) #NEW
    X['Education_score_mu']=temp.mean(axis=1) #NEW
    
    X['Number_uneducated']=temp[temp==1].sum(axis=1).fillna(0)
    X['Education_Number_primary']=temp[temp==2].sum(axis=1).fillna(0)/2
    X['Education_Number_part_secondary']=temp[temp==3].sum(axis=1).fillna(0)/3
    X['Education_Number_secondary']=temp[temp==4].sum(axis=1).fillna(0)/4
    X['Education_Number_college']=temp[temp==5].sum(axis=1).fillna(0)/5
    X['Education_Number_higher']=temp[temp==6].sum(axis=1).fillna(0)/6
    
    col_start="D2_1"
    col_end="D2_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    
    X['MaritalStatus_num_single']=temp[temp==1].sum(axis=1).fillna(0)
    X['MaritalStatus_num_divorced']=temp[temp==2].sum(axis=1).fillna(0)/2
    X['MaritalStatus_num_married']=temp[temp==3].sum(axis=1).fillna(0)/3
    X['MaritalStatus_num_widowed']=temp[temp==4].sum(axis=1).fillna(0)/4
    
    
    X['MaritalStatus_HeadofHH_single']=temp[temp==1].sum(axis=1).fillna(0)
  
    
    
    col_start="D3_1"
    col_end="D3_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    #temp=temp.iloc[:,0:28:2]
    
    X['EmploymentStatus_num_working']=temp[temp==1].sum(axis=1).fillna(0)
    X['Employment_num_underage']=temp[temp==2].sum(axis=1).fillna(0)/2
    X['Employment_num_part_pupil']=temp[temp==3].sum(axis=1).fillna(0)/3
    X['Employment_num_student']=temp[temp==4].sum(axis=1).fillna(0)/4
    X['Employment_num_housewife']=temp[temp==5].sum(axis=1).fillna(0)/5
    X['Employment_num_retired']=temp[temp==6].sum(axis=1).fillna(0)/6
    X['Employment_num_unemployed']=temp[temp==7].sum(axis=1).fillna(0)/7
    
    X['Head_of_Family_position']=d.D4.fillna(7)
    
    X['Family_Income']=d.D6
    
    X['Financial_Status']=d.D7.fillna(d.D7.mean())
    
    #X['socioeconomic_status']=d.D8
    
    temp=d.D9.copy()
    temp[temp>=2]=0
    X['isHouseOwner']=temp.fillna(0)
    
    X['Current_address_Duration']=d.D10
    
    X['isSecondHouse']=d.D11
    X['isWatchSecondHouse']=d.D12.fillna(0)
    
    X['Current_address_duration_per_year']=d.D13
    
    ###
    
    X['SES']=d.SES
    X['ECO_ZONE']=d.ECO_ZONE
    X['City_Name']=d.City_Name
    X['Urb_Rur']=d.Urb_Rur
    X['Cluster']=d.Cluster
    
    ###
    X=X.fillna(0) #just in case
    
    return Y,X

@st.cache_data()    ###suppress_st_warning=True
def summary_stats(d):
    #st.write('INSIDE SUMMARY STATS')
    d = d.apply(pd.to_numeric, errors='coerce') #remove strings
    
    col_start="B19_1"
    col_end="B19_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp=d.iloc[:,idx_start:idx_end+1]
    temp=temp*200
    col_start="B20_1"
    col_end="B20_14"
    idx_start=d.columns.get_loc(col_start)
    idx_end=d.columns.get_loc(col_end)
    temp2=d.iloc[:,idx_start:idx_end+1]
    #temp2=d.iloc[:,idx_start:idx_end+2]
    #temp2=temp2.iloc[:,1:28:2]
    
    from datetime import date
    # creating the date object of today's date
    todays_date = date.today() # to handle year of birth by mistake rather than age
    temp2[temp2>150]=todays_date.year-temp2
    temp2.columns=temp.columns
    temp3=temp2+temp
    
    df = pd.DataFrame(index=range(1))
    df['Male_0_18']=temp3[temp3<=218].count().sum()
    df['Male_19_35']=temp3[(temp3>=219) & (temp3<=235)].count().sum()
    df['Male_36_55']=temp3[(temp3>=236) & (temp3<=255)].count().sum()
    df['Male_55_over']=temp3[(temp3>255) & (temp3<400)].count().sum()
    df['Female_0_18']=temp3[(temp3>=400) & (temp3<=418)].count().sum()
    df['Female_19_35']=temp3[(temp3>=419) & (temp3<=435)].count().sum()
    df['Female_36_55']=temp3[(temp3>=436) & (temp3<=455)].count().sum()
    df['Female_55_over']=temp3[temp3>455].count().sum()
    df['N/A']=temp.count().sum()-temp2.count().sum() if temp.count().sum()-temp2.count().sum()>0  else  0
    df['Total']=temp3.count().sum()+df['N/A']
    
    return df.iloc[-1]
    
