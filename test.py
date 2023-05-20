import numpy as np

a = np.random.standard_normal(size=10)

print(a)

a = np.where(a > 0.4, a - 0.4, np.where(a < -0.4, a + 0.4, 0))

print(a)