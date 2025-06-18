import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")


# plt.scatter(data.studytime, data.score)
# plt.show()

def loss_fn(m, b, points):
    """
    Creating the loss function that will calculate the error and loss.
    We are calculating mean square error

    E = (1/n)Epsilon[y-(mx+b)]^2

    Parameters:
        m: coefficient of input x
        b: bias value
        points: all the data that contains x and y
    Output:
        total_error
    """
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m * x + b))**2
    return total_error / float(len(points))

def gradient_descent(m_now , b_now, points, lr):
    """
    Function to perform gradient descent
    Parameters:
        m_now: current value of coefficient of x
        b_now: current value of y intercept
        points: all the points
        lr: learning rate
    Output:
        returns new m and b values
    """
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2/n)*x*(y- (m_now * x + b_now))
        b_gradient += -(2/n)*(y- (m_now * x + b_now))
    
    m = m_now - m_gradient * lr
    b = b_now - b_gradient * lr

    return m , b

def r_squared(m , b , points):

    actual_y = points.score
    predicted_y = m*points.studytime+b
    mean_y = np.mean(actual_y)
    #Total sum of squares around mean
    ssm = np.sum((actual_y - mean_y)**2)
    #Total sum of squares around fit line
    ssf = np.sum((actual_y - predicted_y)**2)

    if ssm == 0:
        return 1.0

    return 1 - (ssf/ssm)
m = 0
b = 0
lr = 0.001
epochs = 3000

for i in range(epochs):
    loss = loss_fn(m , b , data)
    m, b = gradient_descent(m , b, data, lr)
    if i % 50 == 0:
        print(f"Epoch: {i}\tloss:{loss}")

print(f"m:{m}\tb:{b}")
print(f"R-Squared value: {r_squared(m , b , data)}")

plt.scatter(data.studytime, data.score, label="Data Points", color="blue")

min_x = data.studytime.min()
max_x = data.score.max()

x_line = np.linspace(min_x, max_x, 100) # 100 points for a smooth line
y_line = m * x_line + b

plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label=f'Regression Line: Y = {m:.2f}X + {b:.2f}')
plt.xlabel("Study Time")
plt.ylabel("Score")
plt.xlim(0, 12) # For example, limit x-axis to 0 to 12 hours
plt.ylim(0, 100) # If scores are out of 100
plt.show()
