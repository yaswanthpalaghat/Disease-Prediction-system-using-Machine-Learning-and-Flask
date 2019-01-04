from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import csv,numpy as np,pandas as pd
import os

data = pd.read_csv(os.path.join("templates", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)
#print ("Acurracy: ", clf_dt.score(x_test,y_test))

# with open('templates/Testing.csv', newline='') as f:
#         reader = csv.reader(f)
#         symptoms = next(reader)
#         symptoms = symptoms[:len(symptoms)-1]

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms,indices))

def dosomething(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1,1)).transpose()
    return(dt.predict(user_input_label))

# print(dosomething(['headache','muscle_weakness','puffy_face_and_eyes','mild_fever','skin_rash']))
# prediction = []
# for i in range(7):
#     pred = dosomething(['headache'])   
#     prediction.append(pred) 
# print(prediction)