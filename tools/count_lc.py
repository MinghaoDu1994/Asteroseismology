import numpy as np
import os
import shutil
import matplotlib.pyplot as plt 
from Asteroseismology.io import traverse_dir, mkdir
from Asteroseismology.globe import sep
__all__ = ["count_lc_files"]

def count_lc_files(filepath: str, move_file: bool=False):
	'''
	To count light curve files(.fits) of every target

	Input:
	filepath: File path

	Output:
	1. Csv file contains star id and corresponds file number
	2. Figure: histogram distribution
	'''

	names = []
	for root, dirs, files in os.walk('%s' %filepath):
		#print(files)
		continue
	for i in files:
		if i.split('.',1)[1] == 'fits':
			names.append(i.split('-',1)[0].split('r',1)[1])

	kic = np.array(names, dtype=int)
	output_arr = np.zeros((len(set(kic)),2))
	for i in range(len(np.unique(kic))):
		output_arr[i,0] = np.unique(kic)[i]
		output_arr[i,1] = list(kic).count(output_arr[i,0])

	output_arr.astype(np.int)
	np.savetxt('%s/light_curve_counts.txt' %filepath, output_arr, fmt=["%-10d","%-2d"], comments="#", 
		header='KicID,  Number')
	plt.hist(output_arr[:,1], bins=40)
	plt.savefig('%s/Counts_.png' %filepath)

	if move_file == True :
		for name in output_arr[:,0]:
			#
			strr = filepath+'%d/LCs/' %name
			mkdir(strr)

		fn, fp = traverse_dir(filepath, 'fits')
		for i in range(len(fn)):
			new_f = np.int(fn[i].split('-',1)[0].split('r',1)[1])
			shutil.move(fp[i], filepath+str(new_f)+sep+'LCs'+sep+fn[i])

		return "Done for moving files"

	else:
		return "Done for counting files"

































