import numpy as np 
import pandas as pd 
import os

__all__ = ["split_file_to_nrows", "Conclude_Paras"]

def split_file_to_nrows(file=str, nrows=int):
	if os.path.exists('split_files') == False:
		os.mkdir('split_files')
	data = pd.read_csv(file)
	rows = data.shape[0]
	files_counts = np.int(np.ceil(rows/nrows))
	for i in range(1,files_counts):
		new_d = data.iloc[(i-1) * nrows: i * nrows - 1]
		new_d.to_csv('./split_files/_s%s.csv' %i, header=None, index=None)
	last_file = data.iloc[(files_counts-1) * nrows :]
	last_file.to_csv('./split_files/_s%s.csv' %files_counts, header=None, index=None)
	print('Split Done')

def Conclude_Paras(fp, kicid, fn):
	data = np.loadtxt(fp+'%s_%s' %(kicid,fn))
	return data
