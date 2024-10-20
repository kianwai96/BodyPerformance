## Body Performance Classification Project

This project's objective is to create a machine learning model that reliably forecasts a person's physical attributes-based body performance. 
The model will employ classification techniques to group people into discrete classes according to their performance level after being trained on a dataset of body performance data. 
To guarantee the correctness and efficacy of the model, a range of performance criteria will be employed for evaluation. Next, based on each user's expected performance level, the model will be integrated into the web application to 
predict the body performance of the users.

Data source is from https://www.kaggle.com/datasets/kukuroo3/body-performance-data

data shape : (13393, 12)

age : 20 ~64
gender : F,M
height_cm : (If you want to convert to feet, divide by 30.48)
weight_kg
body fat_%
diastolic : diastolic blood pressure (min)
systolic : systolic blood pressure (min)
gripForce
sit and bend forward_cm
sit-ups counts
broad jump_cm
class : A,B,C,D ( A: best) / stratified
