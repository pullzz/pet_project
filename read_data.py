#!/usr/bin/env python2

import pandas as pd
import numpy as np

def get_users(data_file):

	data = pd.read_pickle(data_file)

	# shuffle the data order
	data = data.sample(frac=1).reset_index(drop=True)

	n_cat = 0 
	cat_owners = []
	n_dog = 0
	dog_owners = []
	total_com = 0
	creators = []
	users = []
	others = []
	cat_dog = []
	for x in range(len(data["User"])):

		total_com += 1

		c_own = own_cat(data["Comment"][x])
		n_cat += c_own

		d_own = own_dog(data["Comment"][x])
		n_dog += d_own

		if c_own == 1 and d_own == 0:
			cat_owners.append(data["User"][x])			

		elif d_own == 1 and c_own == 0:
			dog_owners.append(data["User"][x])	

		elif c_own == 0 and d_own == 0:
			others.append(data["User"][x])

		else:
			cat_dog.append(data["User"][x])

		users.append(data["User"][x])
		creators.append(data["Channel Title"][x])


	cat_owners = list(dict.fromkeys(cat_owners))	
	dog_owners = list(dict.fromkeys(dog_owners))
	others = list(dict.fromkeys(others))
	cat_dog = list(dict.fromkeys(cat_dog))
	users = list(dict.fromkeys(users))	
	creators = list(dict.fromkeys(creators))	

	print "#########################################################################"
	print "Number of cat owners: ", n_cat
	print
	print "Number of dog owners: ", n_dog
	print
	print "Number of cat exclusive owners: ", len(cat_owners)
	print
	print "Number of dog exclusive owners: ", len(dog_owners)
	print
	print "Number of cat and dog owners: ", len(cat_dog)
	print
	print "Number of non-pet owners: ", len(others)
	print
	print "Number of users: ", len(users)
	print
	print "Number of creators: ", len(creators)
	print 
	print "Number of comments: ", total_com
	print "#########################################################################"
	print


	return users, creators, cat_owners, dog_owners, others


def own_cat(cstr):

	own = 0
	k_phrase = ['I own a cat ','I own cats ','I have a cat ','I have cats ',' my cat ','My cat ',' my cats ','My cats ',' our cat ',' our cats ','Our cat ','Our cats ']

	for x in range(len(k_phrase)):
		if k_phrase[x] in cstr:
			own = 1
			break
	return own 

def own_dog(cstr):

	own = 0
	k_phrase = ['I own a dog ','I own dogs ','I have a dog ','I have dogs ',' my dog ','My dog ',' my dogs ','My dogs ',' our dog ',' our dogs ','Our dog ','Our dogs ']

	for x in range(len(k_phrase)):
		if k_phrase[x] in cstr:
			own = 1
			break
	return own 


if __name__=='__main__':

	main()
