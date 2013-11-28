import os
import csv

cur_dir = os.getcwd()
source_path = cur_dir + "/../data/result/prediction.csv"
dst_path = cur_dir +"/../data/result/submission.csv"
if (os.path.exists(dst_path)):
	os.remove(dst_path)
reader = csv.reader(file(source_path))
writer = csv.writer(file(dst_path, 'a'))

header = ["id","s1","s2","s3","s4","s5","w1","w2","w3","w4","k1","k2","k3","k4","k5","k6","k7","k8","k9",\
	"k10","k11","k12","k13","k14","k15"]
content = []
normalized_content = []

for item in reader:
	content.append(item)

length = len(content)
# id
for item in content:
	normalized_content.append([item[0]])
# attrs
for i in xrange(0, length):
	# attitude
	for j in xrange(1, 6):
		num = float(content[i][j])
		if (num > 1):
			normalized_content[i].append(1)
		elif (num > 0.05):
			normalized_content[i].append(num)
		else:
			normalized_content[i].append(0)
	summary = 0
	for j in xrange(1, 6):
		summary += normalized_content[i][j]
	if (summary != 0):
		for j in xrange(1, 6):
			normalized_content[i][j] /= summary

	# time
	for j in xrange(6, 10):
		num = float(content[i][j])
		if (num > 1):
			normalized_content[i].append(1)
		elif (num > 0.05):
			normalized_content[i].append(num)
		else:
			normalized_content[i].append(0)
	summary = 0
	for j in xrange(6, 10):
		summary += normalized_content[i][j]
	if (summary != 0):
		for j in xrange(6, 10):
			normalized_content[i][j] /= summary

	# weather
	for j in xrange(10, 25):
		num = float(content[i][j])
		if (num > 0):
			normalized_content[i].append(num)
		else:
			normalized_content[i].append(0)

writer.writerow(header)
writer.writerows(normalized_content)