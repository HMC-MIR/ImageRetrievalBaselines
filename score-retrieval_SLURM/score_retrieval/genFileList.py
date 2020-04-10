import os
#jpg_dir = "/pylon5/ir5phqp/dhyang/tmp_dir_imageExtract5"
jpg_dir = "/pylon5/ir5phqp/dhyang/ref_jpgs"
outfile = "/home/dhyang/fileList_ref.txt"
f = open(outfile, 'a')
for root, dirs, files in os.walk(jpg_dir):
	for fname in files:
		fpath = os.path.join(root,fname)
		ext = os.path.splitext(fpath)[1]
		print(fpath)
		if ext == ".jpg" or ext == ".png":
			print(fpath, file = f)
f.close()	
