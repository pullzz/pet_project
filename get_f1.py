#!/usr/bin/env python2

import pandas as pd
import pickle
import numpy as np
import random
from bitarray import bitarray

from pyspark.ml.linalg import Vectors
from read_data import get_users

def features(data_file, users, creators, cat_owners, dog_owners, others):

	print "Loading observations for feature set 1..."
	print

	ob_dict = observations_1(data_file, users, creators)

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

		fmat[x] = ob_dict[users[x]]

		# to make sure all classes have the same shuffling
		if users[x] in cat_owners:
			c_fmat[c] = ob_dict[users[x]]
			r_cat.append(users[x])
			c+=1
		elif users[x] in dog_owners:
			d_fmat[d] = ob_dict[users[x]]
			r_dog.append(users[x])
			d+=1
		else:
			o_fmat[o] = ob_dict[users[x]]
			r_others.append(users[x])
			o +=1

	return c_fmat, d_fmat, o_fmat

def observations_1(data_file, users, creators):

	obs = dict((str(users[x]), Vectors.sparse(len(creators),[],[])) for x in range(len(users)))
	data = pd.read_pickle("all_comments.pkl")

	i = 0
	for x in range(len(data["Channel ID"])):

		i+=1
		if i > 1: # skip first row with column names

			c_name = data["Channel Title"][x]
			u_name = data["User"][x]

			c_id = creators.index(c_name)

			f_tmp = obs[str(u_name)]		
			idx = f_tmp.indices
			val = f_tmp.values

			if c_id not in idx:
				idx = np.append(idx,c_id)
				idx.sort()
				val = np.append(val,1)

			obs[u_name] = Vectors.sparse(len(creators),idx,val)

	return obs
