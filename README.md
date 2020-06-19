# Disaster_Pipeline
A machine learning pipeline built to categorize emergency messages based on the needs communicated by the sender

## Installation 
All the code are based on Anaconda distribution of Python. The code should run with no issues using Python versions 3.*. Required packages include: numpy, pandas, json, sklearn(0.19.1), nltk(3.2.5), flask, plotly, sqlalchemy.

To install packages of a specific version, use command:

pip install package_name = version, i.e., pip install scikit-learn = 0.19.1

## Project Motivation
This is a project for the Data Scientist Nanodegree in Udacity (Project #3).

The motivation of this project is to build tool/classifier to categorize tweets/news/messages of disaster time to certain categories to facilitate identifying help needed or emergency situation. For example, if someone asks for help in a hurricane, this classifier should be able to identify it and then atrribute the information to helpers. 

For this purpose, this tool/classifier need to 1) filter out unrelated messages, 2) identify related messages and 3) categorize related messages to certain topics. 

## File Descriptions 
There are three folders:

data: 
- disaster_categories.csv, disaster_messages.csv   | raw data that contains messages and categories they belong to
- process_data.py   | ETL pipline to clean and save to a database file
- DisasterResponse.db   | cleaned data

model: 
- train_classifier.py   | script to build a classfier based on data
- classifier.pkl   | Constructed classifier

app: 
- run.py   | running this script will deploy a web app that could categorize any message you put in
- templates | web design related files 


## Results 
To use the classifier, go to the app folder and run:

python run.py

If you're using your local machine, be sure to change the host from 0.0.0.0 to 127.0.0.1, which I've marked in the script. 

