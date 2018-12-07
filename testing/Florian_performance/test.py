import timeit


a="Hallo Welt"

def testf():
  print a
  
zeit=timeit.timeit(testf,number=1000000)

print zeit