#PHI LAM
#COEN 140 HOMEWORK 2
#qda_classifier.py

################################# IMPORTS ###################################
import numpy as np
import math
import sys
from operator import itemgetter

########################## VARIABLE DECLARATIONS ###########################
#   Elements:
#       1. sepal length in cm
#       2. sepal width in cm
#       3. petal length in cm
#       4. petal width in cm
#       5. class: Iris-setosa, Iris-versicolour, Iris-virginica
#
#   Structure: data[0] = [sep_len, sep_wid, pet_len, pet_wid, "Iris-setosa"]
#              In math, represented by column vector
#
#   First 40 of each class -> training data
#   Last 10 of each class -> test data

training_cols = 5        #five features
training_rows = 120      #80% of one class for classification (40 of 50), 3 classes
test_cols = 5
test_rows = 30          #20% of one class (10 of 50), 3 classes
class_rows = 40
class_cols = 4
source_training_data = [[None for x in range(training_cols)] for y in range(training_rows)]
source_test_data = [[None for x in range(test_cols)] for y in range(test_rows)]

#----Split up training data into the 3 classes----
training_data = np.asarray(source_training_data)
test_data = np.asarray(source_test_data)
class1_training = np.zeros((class_rows, class_cols))
class2_training = np.zeros((class_rows, class_cols))
class3_training = np.zeros((class_rows, class_cols))
class1_test = np.zeros((test_rows, class_cols))
class2_test = np.zeros((test_rows, class_cols))
class3_test = np.zeros((test_rows, class_cols))
class1_mu = 0
class2_mu = 0
class3_mu = 0
class1_covariance = np.zeros((4, 4))
class2_covariance = np.zeros((4, 4))
class3_covariance = np.zeros((4, 4))

#---For debugging---
testing = False
testing_covariance = False
testing_populate = False
testing_main = False
testing_calculate_qda = False
########################## FUNCTION DEFINITIONS #############################

#-----------------------------------------------------------------
# FUNCTION: calculate_covariance()
#	Description: given a 1x4 mu vector, calculate the covariance by
#			calculating the sum of the outer products. The result is
#			a 4x4 matrix.
#	Notes: (in the derviation, x is a column vector, but we represent
#			x here as a row vector instead)
#
def calculate_covariance(training_class, mu_vector):
	covariance = np.zeros((4, 4))
	for index, row in enumerate(training_class):
		temp = row - mu_vector
		covariance += np.outer(temp, temp)
		#incorrect? #covariance += np.outer(temp, np.transpose(temp))
		#if testing_covariance: print("covariance row: ", row)
		#if testing_covariance: print("covariance iteration: ", covariance)
#	if testing_covariance: print("covariance: ", covariance) #debugging
	covariance = (1/class_rows) * covariance				#1/N
	return covariance

#-----------------------------------------------------------------
# FUNCTION: calculate_mu()
# 	Description: Calls np.mean to calculate the mean of each feature
#					(calculates by column)
# 	Notes:
#		p = 4; mu is 1x4 matrix: [a, b, c, d]
#		(mathematically, mu is a column vector)
#
def calculate_mu(np_array):
	mu_vector = np.mean(np_array, axis = 0)
	return mu_vector

#------------------------------------------------------------------
# FUNCTION: calculate_probability()
#
#	x and mu are vectors
#	If called by lda_classify(), will receive the mean covariance
#
def calculate_probability(mu, covariance, x):
	if testing_calculate_qda: print("Inside calculate_probability_qda: covariance = ")
	if testing_calculate_qda: print(covariance)
	if testing_calculate_qda: print("x = ", x)
	if testing_calculate_qda: print("mu = ", mu)

	np.subtract(x, mu)
	term1 = np.multiply(math.sqrt(math.pow((2 * math.pi), 4)), np.linalg.det(covariance))

	term_exp1 = np.multiply(-1.0, np.transpose(np.subtract(x, mu)))
	term_exp2 = np.matmul(np.linalg.inv(covariance), np.subtract(x, mu))

	if testing_calculate_qda: print("term_exp1 = ", term_exp1)
	if testing_calculate_qda: print("term_exp2 = ", term_exp2)

	term_exp3 = np.matmul(term_exp1, term_exp2)
	if testing_calculate_qda: print("term_exp3 = ", term_exp3)

	term2 = math.exp(term_exp3)

	probability = (1/term1) * term2
	probability = probability * 1/3		#prior probability p(c)

	return probability

#------------------------------------------------------------------
# FUNCTION: diag_covariance(covariance_matrix)
#
#	Description: Convert a given covariance matrix into a diagonal
#				 matrix.
def diag_covariance(covariance_matrix):
	diag = np.zeros((4,4))
	for i in range(covariance_matrix.shape[0]):
		diag[i, i] = covariance_matrix[i, i]
	return diag

#-----------------------------------------------------------------
# FUNCTION: populate()
#   Input: dataset.txt
#	Procedure:
#		1) Take first 40 entries of one class and place in training_data
#		2) Take remaining 10 entries of that class and place in test_data
#	Notes:
	#	training[0-39]: iris-setosa
	# 	training[40-79]: iris-versicolor
	#	training[80-119]: iris-virginica
	#	test[0-9]: iris-setosa
	#	test[10-19]: iris-versicolor
	#	test[20-29]: iris-virginica
def populate(in_file):
	global training_data, test_data
	global class1_training, class2_training, class3_training
	global class1_test, class2_test, class3_test

	for sampleid, line in enumerate(in_file):		#Row: one flower's features
		features = line.split(",")
		for featureid, feature in enumerate(features):
			feature = feature.strip()

			#---Populate training_data[][]---
			if sampleid < 40:									#training[0-39]: iris-setosa
				source_training_data[sampleid][featureid] = feature
			elif sampleid >= 50 and sampleid < 90:				#training[40-79]: iris-versicolor
				source_training_data[sampleid-10][featureid] = feature
			elif sampleid >= 100 and sampleid < 140:			#training[80-119]: iris-virginica
			 	source_training_data[sampleid-20][featureid] = feature

			#---Populate test_data[][]---
			elif sampleid >= 40 and sampleid < 50:				#test[0-9]: iris-setosa
				source_test_data[sampleid-40][featureid] = feature
			elif sampleid >= 90 and sampleid < 100:				#test[10-19]: iris-versicolor
				source_test_data[sampleid-80][featureid] = feature
			elif sampleid >= 140:								#test[20-29]: iris-virginica
				source_test_data[sampleid-120][featureid] = feature
			else:
				break

	#---Convert training data and test data to numpy arrays, remove string category---
	training_data = np.asarray(source_training_data)
	test_data = np.asarray(source_test_data)

	training_data = np.delete(training_data, 4, 1)
	test_data = np.delete(test_data, 4, 1)

	training_data = training_data.astype(float)
	test_data = test_data.astype(float)

	#----------Place training data in its own classes-------------
	class1_training = training_data[0:39]
	class2_training = training_data[40:79]
	class3_training = training_data[80:119]
	class1_test = test_data[0:9]
	class2_test = test_data[10:19]
	class3_test = test_data[20:29]

	#-----Debugging-------
	if testing_populate: print("training data: ", training_data)
	if testing_populate: print("test data: ", test_data)
	if testing_populate: print("class1 training: ", class1_training)
	if testing_populate: print("class1 test: ", class1_test)

#-----------------------------------------------------------------
# FUNCTION: predict()
#
#	Description: depends on multiple global variables:
#		- source_test_data
#		- class1_mu, class2_mu, and class3_mu
#		- class1_covariance, class2_covariance, and class 3_covariance
#
def predict(p1, p2, p3, classification):
	match = False

	if p1 > p2 and p1 > p3:
		prediction = "Iris-setosa"

	elif p2 > p1 and p2 > p3:
		prediction = "Iris-versicolor"

	elif p3 > p1 and p3 > p2:
		prediction = "Iris-virginica"

	else:	#no specific match
		print("---No match---")
		print("p1: ", p1)
		print("p2: ", p2)
		print("p3: ", p3)

	if prediction == classification:
		match = True
		#print("Match!")
	else:
		print("*** Failed match ***")
		print("Prediction: ", prediction)
		print("Actual: ", classification, "\n")

	return match

#-----------------------------------------------------------------
# FUNCTION: lda_classify()
#
#	Description: depends on multiple global variables:
#		- source_test_data
#		- class1_mu, class2_mu, and class3_mu
#		- class1_covariance, class2_covariance, and class 3_covariance
#
def lda_classify(test_set_matrix, test_set_list):
	matches = 0
	p1 = 0
	p2 = 0
	p3 = 0

	#-----Calculate average covariance, used for calculating all probalities.
	premean_sum = class1_covariance + class2_covariance + class3_covariance
	mean_covariance = np.divide(premean_sum, 3)

	for i, row in enumerate(test_set_list):
		#print("LDA Classifying: ", row[0:4])
		p1 = calculate_probability(class1_mu, mean_covariance, test_set_matrix[i])
		p2 = calculate_probability(class2_mu, mean_covariance, test_set_matrix[i])
		p3 = calculate_probability(class3_mu, mean_covariance, test_set_matrix[i])

		if predict(p1, p2, p3, row[4]) == True:
			matches += 1

	print("correct matches = ", matches)
	print("possible correct = ", i+1)
	return matches

#-----------------------------------------------------------------
# FUNCTION: qda_classify()
#
#	Description: depends on multiple global variables:
#		- source_test_data
#		- class1_mu, class2_mu, and class3_mu
#		- class1_covariance, class2_covariance, and class 3_covariance
#
def qda_classify(test_set_matrix, test_set_list):
	matches = 0
	p1 = 0
	p2 = 0
	p3 = 0

	for i, row in enumerate(test_set_list):
		#print("QDA Classifying: ", row[0:4])
		p1 = calculate_probability(class1_mu, class1_covariance, test_set_matrix[i])
		p2 = calculate_probability(class2_mu, class2_covariance, test_set_matrix[i])
		p3 = calculate_probability(class3_mu, class3_covariance, test_set_matrix[i])

		if predict(p1, p2, p3, row[4]) == True:
			matches += 1

	print("correct matches = ", matches)
	print("possible correct = ", i+1)
	return matches


################################# main ###################################

if len(sys.argv) == 2:
	try:
		f1 = open(sys.argv[1])
	except:
		print("Usage: arguments must be text files")
		exit()
elif len(sys.argv) == 3:
	try:
		f1 = open(sys.argv[1])
		sys.stdout = open(sys.argv[2], "w")
	except:
		print("Usage: arguments must be text files")
		exit()
else:
	print("Usage: iris_classifier.py dataset.txt out_file.txt")
	print("Note: if no output file, will print to stdout")
	exit()

#----Take input data-----
print("--- Processing training data ---")
populate(f1)
print("--- Done ---")

#----Determine parameters mu and covariance
class1_mu = calculate_mu(class1_training)
class2_mu = calculate_mu(class2_training)
class3_mu = calculate_mu(class3_training)
class1_covariance = calculate_covariance(class1_training, class1_mu)
class2_covariance = calculate_covariance(class2_training, class2_mu)
class3_covariance = calculate_covariance(class3_training, class3_mu)

# print(class1_mu)
# print(class2_mu)
# print(class3_mu)
# print(class1_covariance)
# print(class2_covariance)
# print(class3_covariance)

#------Begin LDA Classification
print("==========================")
print("LDA Classifying: Test Data")
print("==========================")
matches = lda_classify(test_data, source_test_data)
#print("---------- LDA ERROR ---------")
print("LDA Error = ", 1 - matches/test_rows, "\n")

print("==============================")
print("LDA Classifying: Training Data")
print("==============================")
matches = lda_classify(training_data, source_training_data)
#print("---------- LDA ERROR ---------")
print("LDA Error = ", 1 - matches/training_rows, "\n")

#-----Begin QDA classification
print("==========================")
print("QDA Classifying: Test Data")
print("==========================")
matches = qda_classify(test_data, source_test_data)
#print("---------- QDA ERROR ---------")
print("QDA Error = ", 1 - matches/test_rows, "\n")

print("==============================")
print("QDA Classifying: Training Data")
print("==============================")
matches = qda_classify(training_data, source_training_data)
#print("---------- QDA ERROR ---------")
print("QDA Error = ", 1 - matches/training_rows, "\n")

#------Test feature removal

#------Assume independent features, run test again
print("====================================")
print("RETEST ASSUMING INDEPENDENT FEATURES")
print("====================================")
class1_covariance = diag_covariance(class1_covariance)
class2_covariance = diag_covariance(class2_covariance)
class3_covariance = diag_covariance(class3_covariance)

#------Begin LDA Classification
print("==========================")
print("LDA Classifying: Test Data")
print("==========================")
matches = lda_classify(test_data, source_test_data)
#print("---------- LDA ERROR ---------")
print("LDA Error = ", 1 - matches/test_rows, "\n")

print("==============================")
print("LDA Classifying: Training Data")
print("==============================")
matches = lda_classify(training_data, source_training_data)
#print("---------- LDA ERROR ---------")
print("LDA Error = ", 1 - matches/training_rows, "\n")

#-----Begin QDA classification
print("==========================")
print("QDA Classifying: Test Data")
print("==========================")
matches = qda_classify(test_data, source_test_data)
#print("---------- QDA ERROR ---------")
print("QDA Error = ", 1 - matches/test_rows, "\n")

print("==============================")
print("QDA Classifying: Training Data")
print("==============================")
matches = qda_classify(training_data, source_training_data)
#print("---------- QDA ERROR ---------")
print("QDA Error = ", 1 - matches/training_rows, "\n")







#if testing: print("------Training Data-----\n",training_data)
#if testing: print("------Test Data-----\n", test_data)
#if testing: print("-----Source Training Data-----\n", source_training_data)
#if testing: print("-----Source Test Data-----\n", source_test_data)

#############
####
#### OLD CODE
####
#############
