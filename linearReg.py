from numpy import *
import matplotlib.pyplot as plt

def run():
	#gen from text is library from numpy which is used to generate data points and save it to memory.
	points = genfromtxt('data.csv',delimiter = ',')

	#initialize parameters for h(x) = mx + b where m and b are parameters
	m_init = 0;
	b_init = 0;

	#Now we calculate gradient descent and tune parameters m and b
	[m,b] = find_optimal_values_by_gradient_descent(m_init,b_init,array(points))
	print([m,b])
	x_vals = [];
	for i in range(len(points)):
		plt.scatter(points[i,0],points[i,1])
		x_vals.append(points[i,0])
	plt.plot(x_vals, [m*i + b for i in x_vals])
	plt.show()


def find_optimal_values_by_gradient_descent(m_init,b_init,points):
	#Lets define no of iterations and the learning rate
	no_of_iterations = 1000
	learning_rate = 0.0001
	m = m_init
	b = b_init
	#we will calculate cost function(J) which basically is error between actual y and our predicted y for every point and square this term which yields mean square error
	#out parameter a will be (a-(learning_rate*(1/m))*[(partial derivative wrt a(J)*(x)])
	#Note that we will update all parameters simultaneously
	for i in range(len(points)):
		temp_b = b - ((learning_rate)/float(len(points)))*(calculate_derivative(m,b,points,False))
		temp_m = m - ((learning_rate)/float(len(points)))*(calculate_derivative(m,b,points,True))
		b = temp_b
		m = temp_m

	return [m,b]


def calculate_derivative(m,b,points,x_i):
	derivative = 0;
	for i in range(len(points)):
		if x_i:
			derivative = derivative + (m*points[i,0] + b - points[i,1])*points[i,0]
		else:
			derivative = derivative + (m*points[i,0] + b - points[i,1])

	return derivative


if __name__ == '__main__':
	run()