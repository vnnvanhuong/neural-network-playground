# f(x) = x^2 + 2x + 5

x=6
lr=0.01
precision=0.0001
epochs=1000

def f(x):
	return x**2 + 2*x + 5

def df(x):
	return 2*x + 2

for i in range(epochs):
	x0 = x
	x = x - lr*df(x)
	loss = abs(x0 - x)
	print("--> Epoch {0} - loss: {1}".format(i, loss))

	if(loss <= precision):
		break

print("Minimum at {0} - value: {1}".format(x, f(x)))