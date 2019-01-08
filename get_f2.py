#!/usr/bin/env python2

import pickle
import numpy as np
import random

import os
import csv
from bitarray import bitarray
from pyspark.ml.linalg import Vectors

import pandas as pd
from langdetect import detect
import operator

def features(users,creators,cat_owners,dog_owners,others,data_file):

	##########################################################
	# Load randomized users
	##########################################################

	print "Loading observations for feature set 2..."
	print

	ob_dict = observations_2(users,creators,data_file)

	fmat = [[] for x in range(len(users))]
	c_fmat = [[] for x in range(len(cat_owners))]
	d_fmat = [[] for x in range(len(dog_owners))]
	o_fmat = [[] for x in range(len(others))]

	r_cat = []
	r_dog = []
	r_others = []
	c = 0
	d = 0
	o = 0
	for x in range(len(users)):

		fmat[x] = ob_dict[str(users[x])]		

		# to make sure all classes have the same shuffling
		if users[x] in cat_owners:
			c_fmat[c] = ob_dict[str(users[x])]
			r_cat.append(users[x])
			c+=1
		elif users[x] in dog_owners:
			d_fmat[d] = ob_dict[str(users[x])]
			r_dog.append(users[x])
			d+=1
		else:
			o_fmat[o] = ob_dict[str(users[x])]
			r_others.append(users[x])
			o +=1

	return c_fmat, d_fmat, o_fmat


def observations_2(users,creators,data_file):

	max_words = None
	eng_comm = None

	# check if max_words file exists if not create it
	if (os.path.isfile("max_words.npy") == False) or (os.path.isfile("en_only.csv") == False):
		eng_comm, max_words = get_fwords_eng(data_file)
	
	if max_words == None:	
		max_words = np.load("max_words.npy")
		with open('en_only.csv','rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			eng_comm = list(reader)

	max_words2 = []
	for x in range(len(max_words)):
		max_words2.append(max_words[x][0])

	ob_dict = get_ob_2(users,max_words2, eng_comm) 

	return ob_dict


def get_ob_2(users,max_words,data):

	obs = dict((str(users[x]), Vectors.sparse(len(max_words),[],[])) for x in range(len(users)))
	for row in data:

		u_name = row[1]
		f_tmp = obs[str(u_name)]		
		ivec = f_tmp.indices
		val = f_tmp.values

		comm = row[2]
		comm = comm.split(' ')
		for x in range(len(comm)):
			if comm[x] in max_words:

				w_idx = max_words.index(comm[x])	

				if w_idx not in ivec:
					ivec = np.append(ivec,w_idx)
					ivec.sort()
					val = np.append(val,1)
		
		obs[str(u_name)] = Vectors.sparse(len(max_words),ivec,val)

	return obs


def get_fwords_eng(data_file):

	print
	print "Filtering for english comments..."
	print

	en_only = []
	word_dict = {}
	data = pd.read_pickle(data_file)
	

	for i in range(len(data["Comment"])):

		if i % 10000 == 0:
			print "Comments Filtered: ", i+1


		#filter for ascii
		t = ""
		for x in range(len(data["Comment"][i])):
			if ord(data["Comment"][i][x]) < 128:
				t = t+data["Comment"][i][x]

		d = ""
		try:
			d = detect(t)
		except:
			pass

		if d == "en":

			t = t.lower()
			en_only.append([data["Channel Title"][i],str(data["User"][i]),t])
			wlist = t.split(" ")
			for x in range(len(wlist)):
				if wlist[x] in word_dict:
					word_dict[wlist[x]] += 1
				else:
					word_dict[wlist[x]] = 1


	print
	print "Getting word frequency..."
	print 

	# get top num words
	num = 1000
	max_words = []
	for x in range(num):
		tmp = max(word_dict.iteritems(), key=operator.itemgetter(1))
		max_words = max_words + [tmp]	
		word_dict.pop(tmp[0])

	# save temporary results
	with open("en_only.csv", "wb") as f:
	    writer = csv.writer(f)
	    for x in range(len(en_only)):
	    	writer.writerow(en_only[x])
	
	np.save("max_words",max_words)

	return [en_only,max_words]
