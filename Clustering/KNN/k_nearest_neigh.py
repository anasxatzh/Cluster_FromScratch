
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import random
from collections import Counter
from sklearn import preprocessing
import time

_useWhat : str = "ggplot"
testSize : float = .2
_path = r".WorkAll\InptData\chronic_kidney_disease.csv"
plt.style.use(_useWhat)


class do_KNN:
	
	def __init__(self,
			  accurate_predictions : int = 0,
			  total_predictions : int = 0,
			  accuracy : float = 0.0,
			  k : int = 3,
			  distributions : list = []):
		self.accurate_predictions, self.total_predictions, self.accuracy, self.k, self.distributions = \
			accurate_predictions, total_predictions, accuracy, k, distributions


	def do_predict(self, 
				training_data, 
				to_predict) -> tuple[int,int]:
		if len(training_data) >= self.k: return "K cannot be smaller than the total voting groups(ie. number of training data points)"

		for group in training_data:
			for feat in training_data[group]:
				self.distributions.append([np.linalg.norm(np.array(feat) - np.array(to_predict)), 
							              group]
										  )
		
		results = [i[1] for i in sorted(self.distributions)[:self.k]]
		result = Counter(results).most_common(1)[0][0]
		confidence = Counter(results).most_common(1)[0][1]/self.k
		return result, confidence


	def test(self, 
		     test_set : list, 
		     training_set : list) -> None:
		for g in test_set:
			for data in test_set[g]:
				(predicted_class,confidence) = self.do_predict(training_set, 
                                                               data)
				if predicted_class == g:self.accurate_predictions += 1
				else:print(f"Wrong classification.\nConfidence:{str(confidence * 100)} and class:{str(predicted_class)}")
				self.total_predictions += 1
		self.accuracy = 100*(self.accurate_predictions/self.total_predictions)
		print(f"\nAcurracy :{str(self.accuracy)}%")

def modify_data(df):
	df.replace("?", -999999, inplace = True)
	
	df.replace("yes", 4, inplace = True)
	df.replace("no", 2, inplace = True)

	df.replace("notpresent", 4, inplace = True)
	df.replace("present", 2, inplace = True)
	
	df.replace("abnormal", 4, inplace = True)
	df.replace("normal", 2, inplace = True)
	
	df.replace("poor", 4, inplace = True)
	df.replace("good", 2, inplace = True)
	
	df.replace("ckd", 4, inplace = True)
	df.replace("notckd", 2, inplace = True)

def do_main():
	global _path, testSize
	df = pd.read_csv(_path)
	modify_data(df)
	dataset = df.astype(float).values.tolist()
	
	#Normalize
	x : np.array = df.values
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled) #Replace with normalized values
	random.shuffle(dataset)

	train_set = {2: [], 4:[]}
	test_set = {2: [], 4:[]}
	
	train_data = dataset[:-int(testSize * len(dataset))]
	test_data = dataset[-int(testSize * len(dataset)):]

	[train_set[r[-1]].append(r[:-1]) for r in train_data]
	[test_set[r[-1]].append(r[:-1]) for r in test_data]

	# #Insert data into the test set
	# for r in test_data:
	# 	test_set[r[-1]].append(r[:-1]) # Append the list in the dict will all the elements of the record except the class

	t1 = time.time()
	knn = do_KNN()
	knn.test(test_set, 
		     train_set)
	t2 = time.time()
	print(f"Total time: {t2-t1}")

if __name__ == "__main__":
	do_main()