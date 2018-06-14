#https://github.com/MostAwesomeDude/java-random

import javarandom

if __name__ == '__main__':
	seed = 123
	print('test of javarandom lib')
	print('seed = ' + str(seed))
	r = javarandom.Random(seed)
	for i in range(10):
		print(str(r.nextGaussian()))
	print('done')