f = open('wids','w')
d = {}
for s in open('vocab.en'):
	s = s.split()
	d[s[0]] = s[1]

for s in open('a'):
	for w in s.split():
		print >>f,d.get(w,'1'),
	print >>f
