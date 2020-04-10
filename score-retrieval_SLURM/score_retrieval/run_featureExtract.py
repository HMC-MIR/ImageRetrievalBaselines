import os
import sys
import subprocess
import time
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from score_retrieval.vec_db import save_veclists 


filepath = "/home/dhyang/Evan_CNNRetrieval/score-retrieval/score_retrieval/fileList.txt"
filepath = "/home/dhyang/Evan_CNNRetrieval/score-retrieval1/score_retrieval/fileList_ref.txt"
maxJobs = 4500
basepath = "/pylon5/ir5phqp/dhyang/tmp_dir_imageExtract5/"
def main():
	count = 0
	fnames = []
	with open(filepath, 'r') as f:
		for fname in f:
			fnames.append(fname.strip())
	print("LOADED FILE LIST")
	currentRunning = []
	idx = 0
	while (idx < len(fnames)):
		subprocess.call(['sbatch','bootlegWrapper.sh',str(idx), str(idx+90)])
		idx = idx+90
	return





if __name__ == "__main__":
	main()
