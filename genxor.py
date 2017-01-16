n = 15
l = len(bin(n)[2::])
t = []

for i in xrange(n+1):
	s = bin(i)[2::]
	k = l-len(s)
	s = "0"*k+s
	s = " ".join(map(str, list(s)))
	if(s.count("1")%2==0):
		t.append(0)
	else:
		t.append(1)

	print s
	# print s.count("1")
s = "\n".join(map(str, t))
print s

