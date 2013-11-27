import os
import csv

cur_dir = os.getcwd()

dst_path = cur_dir + "/../data/result/submission.csv"
if os.path.exists(dst_path):
	os.remove(dst_path)
dst_csv = file(dst_path, "w")
dst_writer = csv.writer(dst_csv)

test_csv = file(cur_dir + "/../data/test.csv")
test_reader = csv.reader(test_csv)

header = ["id","s1","s2","s3","s4","s5","w1","w2","w3","w4","k1","k2","k3","k4","k5","k6","k7","k8","k9",\
	"k10","k11","k12","k13","k14","k15"]
ROW_NUM = 42157

# collect ID
test_arr = []
for item in test_reader:
	test_arr.append(item[0])
del test_arr[0]
result_arr = []
for i in xrange(0, ROW_NUM):
	result_arr.append([])
	result_arr[i] = [test_arr[i]]

# collect ATTRIBUTES
for ATTRIBUTES_GROUP in xrange(0, 3):
	if (ATTRIBUTES_GROUP == 0):
		attrs_csv = file(cur_dir + "/../data/result/attitude_res.csv")
	if (ATTRIBUTES_GROUP == 1):
		attrs_csv = file(cur_dir + "/../data/result/time_res.csv")
	if (ATTRIBUTES_GROUP == 2):
		attrs_csv = file(cur_dir + "/../data/result/weather_res.csv")
	attrs_reader = csv.reader(attrs_csv)
	attrs_arr = []
	for item in attrs_reader:
		attrs_arr.append(item)
	for i in xrange(0, ROW_NUM):
		for attr in attrs_arr[i]:
			num = float(attr)
			if (num > 0):
				result_arr[i].append(num)
			else:
				result_arr[i].append(0)
	attrs_csv.close()

# normalize ATTRIBUTES
for item in result_arr:
	# attitude
	summary = 0
	for ATTRIBUTE_NUM in xrange(0, 5): 
		summary += item[ATTRIBUTE_NUM + 1]
	if (summary != 0):
		for ATTRIBUTE_NUM in xrange(0, 5): 
			item[ATTRIBUTE_NUM + 1] /= summary
	# time
	summary = 0
	for ATTRIBUTE_NUM in xrange(5, 9): 
		summary += item[ATTRIBUTE_NUM + 1]
	if (summary != 0):
		for ATTRIBUTE_NUM in xrange(5, 9): 
			item[ATTRIBUTE_NUM + 1] /= summary
	# weather
	summary = 0
	for ATTRIBUTE_NUM in xrange(9, 24): 
		summary += item[ATTRIBUTE_NUM + 1]
	if (summary != 0):
		for ATTRIBUTE_NUM in xrange(9, 24): 
			item[ATTRIBUTE_NUM + 1] /= summary

# write HEADER
dst_writer.writerow(header)

# write ID and ATTRIBUTES
dst_writer.writerows(result_arr)

dst_csv.close()
