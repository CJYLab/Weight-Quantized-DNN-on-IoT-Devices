file_test=open("cjycjy.txt","r")

file_new=open("cjyreplaced","w+")


print(file_test)


#print(type(file_test.read()))

for line in file_test:
#	words=line.split()
#	print(words)	


	newline=line.replace("variable(["," ").replace("])"," ")
	file_new.write(newline)


#inanai="variable(["

