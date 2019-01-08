#!/usr/bin/env python2

from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import pyspark
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors

import numpy as np
import get_f1
import get_f2
from read_data import get_users

def main():

	###########################################################################
	# Assumes
	#
	# Cat Owners
	# Dog Owners
	# Others (non-pet owners)
	#
	# But that CAT & DOG owners don't exist
	###########################################################################

	data_file = "all_comments.pkl"
	spark = SparkSession.builder.getOrCreate()

	print
	print "Reading data..."
	print
	users, creators, cat_owners, dog_owners, others = get_users(data_file)

	c_fmat, d_fmat, o_fmat = get_f1.features(data_file, users, creators, cat_owners, dog_owners, others)

	c_fmat_2, d_fmat_2, o_fmat_2 = get_f2.features(users, creators, cat_owners, dog_owners, others, data_file)

	# split train and test
	p_train = 80

	n_train = int(len(cat_owners) * p_train/float(100))
	n_test = len(cat_owners) - n_train

	# train
	fc_train = c_fmat[0:n_train]
	fd_train = d_fmat[0:n_train]
	fo_train = o_fmat[0:n_train]

	# train 2
	fc_train_2 = c_fmat_2[0:n_train]
	fd_train_2 = d_fmat_2[0:n_train]
	fo_train_2 = o_fmat_2[0:n_train]

	###########################################################################

	# test
	fc_test = c_fmat[n_train : n_train + n_test]
	fd_test = d_fmat[n_train : n_train + n_test]
	fo_test = o_fmat[n_train : n_train + n_test]

	# test 2
	fc_test_2 = c_fmat_2[n_train : n_train + n_test]
	fd_test_2 = d_fmat_2[n_train : n_train + n_test]
	fo_test_2 = o_fmat_2[n_train : n_train + n_test]

	###########################################################################

	# alt test
	afc_test = c_fmat[n_train : n_train + len(c_fmat)]
	afd_test = d_fmat[n_train : n_train + len(d_fmat)]
	afo_test = o_fmat[n_train : n_train + len(o_fmat)]

	# alt test 2
	afc_test_2 = c_fmat_2[n_train : n_train + len(c_fmat_2)]
	afd_test_2 = d_fmat_2[n_train : n_train + len(d_fmat_2)]
	afo_test_2 = o_fmat_2[n_train : n_train + len(o_fmat_2)]

	###########################################################################

	print "Combining features 1 and features 2 for train and test sets..."
	print

	fc_train_com = concate_sparse(fc_train,fc_train_2)
	fd_train_com = concate_sparse(fd_train,fd_train_2)
	fo_train_com = concate_sparse(fo_train,fo_train_2)

	fc_test_com = concate_sparse(fc_test,fc_test_2)
	fd_test_com = concate_sparse(fd_test,fd_test_2)
	fo_test_com = concate_sparse(fo_test,fo_test_2)

	afc_test_com = concate_sparse(afc_test,afc_test_2)

	afd_test_com = concate_sparse(afd_test,afd_test_2)
	afo_test_com = concate_sparse(afo_test,afo_test_2)


	print "Constructing data frame for training..."
	print
	tmp = []
	for x in range(n_train):
		
		tmp.append(Row(label = int(0), features = fc_train_com[x]))
		tmp.append(Row(label = int(1), features = fd_train_com[x]))
		tmp.append(Row(label = int(2), features = fo_train_com[x]))

	df = spark.createDataFrame(tmp)
	NB = NaiveBayes()
	print "Fitting model..."
	print

	model = NB.fit(df)

	print "Constructing data frame for testing..."
	print
	tmp_test = []
	for x in range(n_test):

		tmp_test.append(Row(label = int(0), features = fc_test_com[x]))
		tmp_test.append(Row(label = int(1), features = fd_test_com[x]))
		tmp_test.append(Row(label = int(2), features = fo_test_com[x]))

	df_test = spark.createDataFrame(tmp_test)
	test_pred = model.transform(df_test)

	l_out = test_pred.select("label").collect()
	p_out = test_pred.select("prediction").collect()

	print
	print "############################################################################"
	print "Training and Testing on Balanced Number of Elements in each Class"
	print "############################################################################"
	print

	print_results(l_out,p_out)

	print
	print "############################################################################"
	print "Training Balanced Number of Elements in each Class, Testing Fraction of"
	print "Elements in Each Class Same as in Data"
	print "############################################################################"
	print

	alt_test = []
	for x in range(len(afc_test_com)):
		alt_test.append(Row(label = int(0), features = afc_test_com[x]))

	for x in range(len(afd_test_com)):
		alt_test.append(Row(label = int(1), features = afd_test_com[x]))

	for x in range(len(afo_test_com)):
		alt_test.append(Row(label = int(2), features = afo_test_com[x]))

	alt_df_test = spark.createDataFrame(alt_test)
	alt_test_pred = model.transform(alt_df_test)

	alt_l_out = alt_test_pred.select("label").collect()
	alt_p_out = alt_test_pred.select("prediction").collect()

	print_results(alt_l_out,alt_p_out)


def print_results(l_out, p_out):

	# calculate the accuracy
	c = 0
	cat = 0
	cat_t = 0
	cat_f = 0

	dog = 0
	dog_t = 0
	dog_f = 0

	other = 0
	other_t = 0
	other_f = 0

	false_c = 0
	false_d = 0
	false_o = 0
	for x in range(len(l_out)):
		if p_out[x][0] == l_out[x][0]:
			c += 1

		# cat owners
		if l_out[x][0] == 0:
			cat += 1
			if p_out[x][0] == 0:
				cat_t += 1
			else:
				cat_f += 1

		if p_out[x][0] != 0 and l_out[x][0] != 0:
			false_c += 1

		# dog owners
		if l_out[x][0] == 1:
			dog += 1
			if p_out[x][0] == 1:
				dog_t += 1
			else:
				dog_f += 1

		if p_out[x][0] != 1 and l_out[x][0] != 1:
			false_d += 1

		# non-pet owners
		if l_out[x][0] == 2:
			other += 1
			if p_out[x][0] == 2:
				other_t += 1
			else:
				other_f += 1

		if p_out[x][0] != 2 and l_out[x][0] != 2:
			false_o += 1

	print
	print "Model Accuracy: ", c / float(len(p_out))
	print
	print "Number of Cat Owners Test Samples: ", cat
	print "Correctly Predicted Cat Owner: ", cat_t
	print "Incorrectly Predicted Cat Owner: ", cat_f
	print
	print "Number of Dog Owners Test Samples: ", dog
	print "Correctly Predicted Dog Owner: ", dog_t
	print "Incorrectly Predicted Dog Owner: ", dog_f
	print
	print "Number of Non-Pet Owners Test Samples: ", other
	print "Correctly Predicted Non-Pet Owner: ", other_t
	print "Incorrectly Predicted Non-Pet Owner: ", other_f
	print

	c_tpr = cat_t / float(cat)
	c_tnr = false_c / float(dog+other)

	d_tpr = dog_t / float(dog)
	d_tnr = false_d / float(cat+other)

	o_tpr = other_t / float(other)
	o_tnr = false_o / float(cat+dog)

	print "Cat Owners TPR (Sensitivity): ", c_tpr
	print "Cat Owners TNR (Specificity): ", c_tnr
	print "Cat Owners FPR (Type 1 Error): ", 1-c_tnr
	print "Cat Owners FNR (Type 2 Error): ", 1-c_tpr
	print
	print "Dog Owners TPR (Sensitivity): ", d_tpr
	print "Dog Owners TNR (Specificity): ", d_tnr
	print "Dog Owners FPR (Type 1 Error): ", 1-d_tnr
	print "Dog Owners FNR (Type 2 Error): ", 1-d_tpr
	print
	print "Non-Pet Owners TPR (Sensitivity): ", o_tpr
	print "Non-Pet Owners TNR (Specificity): ", o_tnr
	print "Non-Pet Owners FPR (Type 1 Error): ", 1-o_tnr
	print "Non-Pet Owners FNR (Type 2 Error): ", 1-o_tpr
	print

def concate_sparse(fmat1,fmat2):

	fmat_com = []
	for x in range(len(fmat1)):
		
		vec1 = fmat1[x]
		vec2 = fmat2[x]

		try:
			v1_size = vec1.size
			v1_idx = vec1.indices.tolist()
			v1_val = vec1.values.tolist()

			v2_size = vec2.size
			v2_idx = vec2.indices.tolist()
			v2_val = vec2.values.tolist()

			num2 = v1_size + v2_size
			tmp_loc = []
			tmp_loc = tmp_loc + v1_idx
			for y in range(len(v2_idx)):
				tmp_loc = tmp_loc + [v1_size + v2_idx[y]]

			tmp_val = []	
			tmp_val = tmp_val + v1_val
			tmp_val = tmp_val + v2_val
		
			fmat_com.append(Vectors.sparse(num2,tmp_loc,tmp_val))

		except:
			# vec1 or vec 2 empty
			pass

	return fmat_com


if __name__=='__main__':

	main()
