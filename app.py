#Step 1 - Define Problem - Building a classification model to predict LinkedIn users.

#Step 2 - Import Packages
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Step 3 - Ingest Data

s = pd.read_csv("social_media_usage.csv")
def clean_sm(x):
    x = np.where(x == 1,
                  1,
                  0)
    return x

dd = {"sm_li": clean_sm(s["web1h"]), 
      "income": np.where(s["income"] <=9,
                         s["income"],
                         np.nan),
      "education": np.where(s["educ2"] <=8,
                            s["educ2"],
                            np.nan),
      "parent": clean_sm(s["par"]), 
      "married": clean_sm(s["marital"]), 
      "female": np.where(s["gender"] == 2,
                         1,
                         0),
      "age": np.where(s["age"] <= 98,
                      s["age"],
                      np.nan)}
ss = pd.DataFrame(data=dd).dropna()

#Step 4 - Exploratory Analysis

#Step 5 - Feature Selection
y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]

x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    stratify = y, 
                                                    test_size = 0.2, 
                                                    random_state = 757)

#Step 6 - Train model
lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)

#Step 7 - Evaluate Model
y_pred = lr.predict(x_test)


st.markdown("<h1 style='text-align: center; color: #017BB6;'>LinkedIn Predictor!</h1>", unsafe_allow_html=True)
st.caption("<h6 style='text-align: center; color: #017BB6;'>Fill in the values below to predict LinkedIn usage</h6>", unsafe_allow_html=True)

#Age Slider
age_help = '''Insert an age between 1 and 118.  
Did you know that the oldest person alive is Lucile Randon at 118 years old?'''
age = st.slider("Age :spiral_calendar_pad:", max_value = 118, help = age_help)
st.write("Your age is", age, ".")

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

#Income Downselect
income_help = '''Choose your income level from the dropdown box'''
income_box = st.selectbox(label = "Income :moneybag:", options = ("<$10,000", "$10,000 - $20,000", "$20,000 - $30,000", 
"$30,000 - $40,000", "$40,000 - $50,000", "$50,000 - $75,000", "$75,000 - $100,000", "$100,000 - $150,000", 
">$150,000"), help = income_help)
if(income_box == "<$10,000"):
    st.write("Your income is <$10,000.")
elif(income_box == "$10,000 - $20,000"):
    st.write("Your income is \$10,000 - \$20,000.")
elif(income_box == "$20,000 - $30,000"):
    st.write("Your income is \$20,000 - \$30,000.")
elif(income_box == "$30,000 - $40,000"):
    st.write("Your income is \$30,000 - \$40,000.")
elif(income_box == "$40,000 - $50,000"):
    st.write("Your income is \$40,000 - \$50,000.")
elif(income_box == "$50,000 - $75,000"):
    st.write("Your income is \$50,000 - \$75,000.")
elif(income_box == "$75,000 - $100,000"):
    st.write("Your income is \$75,000 - \$100,000.")
elif(income_box == "$100,000 - $150,000"):
    st.write("Your income is \$100,000 - \$150,000.")
elif(income_box == ">$150,000"):
    st.write("Your income is >\$150,000.")

if(income_box == "<$10,000"):
    income = 1
elif(income_box == "$10,000 - $20,000"):
    income = 2
elif(income_box == "$20,000 - $30,000"):
    income = 3
elif(income_box == "$30,000 - $40,000"):
    income = 4
elif(income_box == "$40,000 - $50,000"):
    income = 5
elif(income_box == "$50,000 - $75,000"):
    income = 6
elif(income_box == "$75,000 - $100,000"):
    income = 7
elif(income_box == "$100,000 - $150,000"):
    income = 8
elif(income_box == ">$150,000"):
    income = 9

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

#Educ Downselect
educ_help = '''Choose your education level from the dropdown box'''
educ_box = st.selectbox(label = "Education :mortar_board:", options = ("Less than high school", "High school incomplete", "High school graduate", 
 "Some college, no degree", "Two-year associate degree (e.g., AS, AA)", "Four-year degree (e.g., BS, BA)", 
 "Some post-graduate or professional schooling", "Post-graduate or professional degree"), help = educ_help)
if(educ_box == "Less than high school"):
    st.write("Your education level is: less than high school.")
elif(educ_box == "High school incomplete"):
    st.write("Your education level is: high school incomplete.")
elif(educ_box == "High school graduate"):
    st.write("Your education level is: high school graduate.")
elif(educ_box == "Some college, no degree"):
    st.write("Your education level is: some college, no degree.")
elif(educ_box == "Two-year associate degree (e.g., AS, AA)"):
    st.write("Your education level is: two-year associate degree.")
elif(educ_box == "Four-year degree (e.g., BS, BA)"):
    st.write("Your education level is: Four-year degree.")
elif(educ_box == "Some post-graduate or professional schooling"):
    st.write("Your education level is: some post-graduate or professional schooling.")
elif(educ_box == "Post-graduate or professional degree"):
    st.write("Your education level is: post-graduate or professional degree.")

if(educ_box == "Less than high school"):
    educ = 1
elif(educ_box == "High school incomplete"):
    educ = 2
elif(educ_box == "High school graduate"):
    educ = 3
elif(educ_box == "Some college, no degree"):
    educ = 4
elif(educ_box == "Two-year associate degree (e.g., AS, AA)"):
    educ = 5
elif(educ_box == "Four-year degree (e.g., BS, BA)"):
    educ = 6
elif(educ_box == "Some post-graduate or professional schooling"):
    educ = 7
elif(educ_box == "Post-graduate or professional degree"):
    educ = 8

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

#Parent/Married/Gender
col1, col2, col3 = st.columns((1,1,1))
with col1:
   parent_input = st.radio("Are you a parent? :baby:", ("Yes", "No"), help = "If you are a parent, select Yes. Otherwise, select No.")

with col2:
   married_input = st.radio("Are you married? :ring:", ("Yes", "No"), help = "If you are married, select Yes. Otherwise, select No.")

with col3:
   gender_input = st.radio("What is your gender? :female_sign: :male_sign:", ("Female", "Male"), help = "Select your gender.")

if(parent_input == "Yes"):
    parent = 1
else: parent = 0

if(married_input == "Yes"):
    married = 1
else: married = 0

if(gender_input == "Male"):
    female = 0
else: female = 1

yourdata = pd.DataFrame({
    "income": [income],
    "education": [educ],
    "parent": [parent],
    "married": [married],
    "female": [female],
    "age": [age]
})

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

yourdata["prediction_li_user"] = lr.predict(yourdata)
li_user = yourdata.iloc[0, 6]
if(li_user == 1):
    st.markdown("<h3 style='text-align: center; color: #017BB6;'>Based on your inputs, our model predicts that you are a LinkedIn user!</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='text-align: center; color: #017BB6;'>Based on your inputs, our model predicts that you are not a LinkedIn user!</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    if(li_user == 1):
        st.image("https://static.vecteezy.com/system/resources/previews/009/097/186/large_2x/blue-color-white-background-linkedin-design-logo-sign-symbol-free-vector.jpg", width = 250)
    else:
        st.image("https://www.square2marketing.com/hubfs/2018%20Blog%20Post%20Images/NoLinkedIn.png", width = 250)

with col3:
    st.write(' ')

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

st.caption("Do you want to learn how to create apps like this? Visit [Gerogetown University MSBA](https://b.landing.msbonline.georgetown.edu/lp/msba/?utm_source=google&utm_medium=cpc&utm_term=georgetown+msba&utm_campaign=RWC_GTNMSB_MSBA_Search-PPC_Paid+Search_Google_Branded_NULL_Domestic_Brand-Program_NULL_Evergreen&utm_content=MSBA%7CProgram%7CExact%7CObservation&uadgroup=MSBA%7CProgram%7CExact%7CObservation&uAdCampaign=RWC_GTNMSB_MSBA_Search-PPC_Paid+Search_Google_Branded_NULL_Domestic_Brand-Program_NULL_Evergreen&gclid=CjwKCAiA-dCcBhBQEiwAeWidtWEjo5mc7NXpmnHX_4I4sAg5pQDn1xDXYGwmQ4NCtVOhOP4-JtmCMRoC2_sQAvD_BwE&gclsrc=aw.ds)!")

#Dr. Lyon/Mr. Dutt,

 #I hope you enjoyed visiting my LinkedIn predictor app!
 #I have no idea if you'll see this or not,
 #but if you do, enjoy your break and happy holidays!

#Aaron