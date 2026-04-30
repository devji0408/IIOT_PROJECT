import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))

sample = np.array([[1,2,1,3,0]])

prediction = model.predict(sample)

print("Predicted Winner:", prediction)