from datetime import datetime
import numpy as np

start = datetime.now()
print ("start at", start.strftime("%Y-%m-%d %H:%M:%S"))

for i in xrange(1000000):
    np.square(np.zeros((100,100)))
end = datetime.now()
print("end at:", end.strftime("%Y-%m-%d %H:%M:%S"))
print("running for:", str(end-start)[:7])
