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
for ATTRIBUTE_NUM in xrange(0, 24):
	file_name = "res_" + str(ATTRIBUTE_NUM) + ".csv"
	res_attr_csv = file(cur_dir + "/../data/result/" + file_name)
	res_attr_reader = csv.reader(res_attr_csv)
	res_attr_arr = []
	for item in res_attr_reader:
		res_attr_arr.append(item[0])
	for i in xrange(0, ROW_NUM):
		num = float(res_attr_arr[i])
		if (num > 0):
			result_arr[i].append(num)
		else:
			result_arr[i].append(0)
	res_attr_csv.close()

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
