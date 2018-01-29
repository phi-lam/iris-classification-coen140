#PHI LAM
#COEN 140 HOMEWORK 2
#qda_classifier.py

#################### IMPORTS #####################
import numpy as np
import math
import sys
from operator import itemgetter

#################### VARIABLE DECLARATIONS ####################
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

training_cols = 5        #five traits
training_rows = 120      #80% of one class for classification (40 of 50), 3 classes
test_cols = 5
test_rows = 30          #20% of one class (10 of 50), 3 classes
training_data = [[None for x in range(training_cols)] for y in range(training_rows)]
test_data = [[None for x in range(test_cols)] for y in range(test_rows)]
line_index = 0
#---For debugging---
testing = True

#################### FUNCTION DEFINITIONS #####################

#-----------------------------------------------------------------
# FUNCTION: populate()
#   Input: dataset.txt
#	Procedure:
#		1) Take first 40 entries of one class and place in training_data
#		2) Take remaining 10 entries of that class and place in test_data
#
def populate(in_file):
	for sampleid, line in enumerate(in_file):		#Row: one flower's features
		features = line.split(",")
		for featureid, feature in enumerate(features):
			feature = feature.strip()
			#---Populate training_data[][]---
			if sampleid < 40:									#training[0-39]: iris-setosa
				training_data[sampleid][featureid] = feature
			elif sampleid >= 50 and sampleid < 90:				#training[40-79]: iris-versicolor
				training_data[sampleid-10][featureid] = feature
			elif sampleid >= 100 and sampleid < 140:			#training[80-119]: iris-virginica
			 	training_data[sampleid-20][featureid] = feature
			#---Populate test_data[][]---
			elif sampleid >= 40 and sampleid < 50:				#test[0-9]: iris-setosa
				test_data[sampleid-40][featureid] = feature
			elif sampleid >= 90 and sampleid < 100:				#test[10-19]: iris-versicolor
				test_data[sampleid-80][featureid] = feature
			elif sampleid >= 140:								#test[20-29]: iris-virginica
				test_data[sampleid-120][featureid] = feature
			else:
				break
#################################################################
# "main"
#

if len(sys.argv) != 3:
	print("Usage: qda_classifier.py dataset.txt out_file")
try:
	f1 = open(sys.argv[1])
	f2 = open(sys.argv[2], "w")
except:
	print("Usage: arguments must be text files")
	exit()

print("Processing training data...")
populate(f1)
print("Done")

if testing: print(training_data)
if testing: print(test_data)
