from calendar import firstweekday
from cmath import nan
from mimetypes import init
import os
from pyclbr import Function
from pyexpat.model import XML_CQUANT_NONE #vscode extension?
import random
from shutil import move
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import time

# stanisz: To have uniform messages for bad input
def print_bad_input_message():
    print("Bad input! Exiting...")

def NewtonScalar(a, b, c, d, xApprox, maxIter = 10000000000, time_limit = 10000000, value_limit = 100000000):
    start_time = time.time()
    dx = 0.0000000001
    f = np.polynomial.Polynomial([d, c, b, a])
    lowest_so_far = xApprox

    for i in range(maxIter):
        if time.time() - start_time > time_limit:
            break
        if f(lowest_so_far) < value_limit:
            break

        if f(xApprox) < f(lowest_so_far):
            lowest_so_far = xApprox    
        
        firstDerivative = ((3 * a * xApprox ** 2) + (2 * b * xApprox) + c)
        secondDerivative = ((6 * a * xApprox) + (2 * b))

        if(firstDerivative == 0):
            xApprox += random.choice([-dx, dx])

        if(secondDerivative == 0):
            secondDerivative -= np.sign(firstDerivative) * dx

        xApprox -= np.sign(firstDerivative) * abs(firstDerivative / secondDerivative)

    if(math.isnan(xApprox)):
        print("The Newton's method could not yield a proper minimum point (infinity)")

    return lowest_so_far

def NewtonMat(A, b, c, xApprox, maxIter = 10000000000, time_limit = 10000000, value_limit = 100000000):
    start_time = time.time()
    dx = 0.0000000001
    lowest_so_far = xApprox
    value_lowest_so_far = sys.float_info.max

    for i in range(maxIter):
        if time.time() - start_time > time_limit:
            break
        if value_lowest_so_far < value_limit:
            break

        value_current = c + b.transpose() * xApprox + \
            xApprox.transpose() * A * xApprox
        if value_current < value_lowest_so_far:
            lowest_so_far = xApprox
            value_lowest_so_far = value_current

        firstDerivative = np.matrix(b).transpose() + (np.matrix(A) * np.matrix(xApprox).transpose()) + (np.matrix(A).transpose() * np.matrix(xApprox).transpose())
        secondDerivative = (np.matrix(A).transpose() + np.matrix(A))

        if(np.count_nonzero(firstDerivative == dx) == firstDerivative.size):
            SIGN = np.ones([len(xApprox), len(xApprox[0])])
            SIGN = [x * random.choice([-dx, dx]) for x in SIGN]
            xApprox += np.matrix(SIGN)

        if(np.count_nonzero(secondDerivative) == 0):
            print("Error Newton Method: Matrix 'A' Cannot Be A Zero Matrix")
            return nan

        xApprox = (np.matrix(xApprox).transpose() - (np.linalg.inv(secondDerivative) * firstDerivative)).transpose()
    
    return lowest_so_far.transpose()

def gradient(a, b, c, d, x_starting, maxIter = 10000000000, time_limit = 10000000, value_limit = 100000000):

    f = np.polynomial.Polynomial([d, c, b, a])
    f_derivative = f.deriv()
    dx = 0.00000000001
    current_x = x_starting
    lowest_so_far = current_x
    start_time = time.time()

    for i in range(maxIter):
        if time.time() - start_time > time_limit:
            break
        if f(lowest_so_far) < value_limit:
            break
        
        if f(current_x) < f(lowest_so_far):
            lowest_so_far = current_x

        slope = f_derivative(current_x)
        if slope == 0:  # stanisz: Floating precision issues!
            current_x += random.choice([-dx, dx])

        if(math.isnan(current_x)):
            print("Gradient Descent method could not yield a proper minimum point (infinity)")

        step = -0.1 * slope
        current_x += step

    
    return lowest_so_far, f(lowest_so_far)

def gradient2(A, B, c, x_starting, maxIter = 10000000000, time_limit = 10000000, value_limit = 100000000):

    A = np.matrix(A)
    B = np.matrix(B).transpose()  # column vector
    x_starting = np.matrix(x_starting).transpose()  # column vector
    dx = 0.00000000001
    current_x = x_starting
    lowest_so_far = current_x
    value_lowest_so_far = sys.float_info.max
    start_time = time.time()

    intermediate_results_x = []  # Plotting
    intermediate_results_y = []

    for i in range(maxIter):
        if time.time() - start_time > time_limit:
            break
        if value_lowest_so_far < value_limit:
            break
        
        value_current = c + B.transpose() * current_x + \
            current_x.transpose() * A * current_x
        intermediate_results_x.append([x.item() for x in current_x])
        intermediate_results_y.append(value_current)

        if value_current < value_lowest_so_far:
            lowest_so_far = current_x
            value_lowest_so_far = value_current

        if 0:
            slope = []

            for outer_i, x_i in enumerate(current_x):

                move_vector = np.zeros(len(current_x))
                move_vector[outer_i] = dx
                move_vector = np.matrix(move_vector).transpose()
                xi_moved = current_x + move_vector
                value_xi_moved = c + B.transpose() * xi_moved + xi_moved.transpose() * A * xi_moved
                # stanisz: From the definition of a gradient
                partial_xi = (value_xi_moved - value_current) / dx

                slope.append(partial_xi.item())

            slope = np.matrix(slope).transpose()  # column vector
            
        else:
            slope = B + A * current_x + A.transpose() * current_x

        step = -0.1 * slope
        current_x += step

    return lowest_so_far, value_lowest_so_far, intermediate_results_x, intermediate_results_y


def main():
    
    os.system("cls")

    print("Welcome to gradient-py!")

    if 1:
        user_config = {
            "iterationsLimit" : None,
            "coefficients" : [],
            "startingPoint" : None
        }

        StoppingConditionSelection = input(
            "Enter C to enter configuration or N for default settings ")

        if StoppingConditionSelection == "C":
            user_config["iterationsLimit"] = int(input("How many iterations is the computational limit? "))
        else:
            user_config["iterationsLimit"] = 100

        FunctionSelection = input(
            "To compute F(x) enter F, to compute G(x) enter G: ")

        if FunctionSelection == "F":
            # stanisz: Computing F
            a = float(input("Enter the scalar value of coefficient \'a\': "))
            b = float(input("Enter the scalar value of coefficient \'b\': "))
            c = float(input("Enter the scalar value of coefficient \'c\': "))
            d = float(input("Enter the scalar value of coefficient \'d\': "))

            user_config["coefficients"].append({"a" : a, "b" : b, "c" : c, "d" : d})

            StartingPointSelection = input(
                "If you would like to enter initial value of \'x\' manually enter M. Enter \'A\' for automatic choice: ")
            
            if StartingPointSelection == "M":
                # stanisz: Manual starting point selection
                initial_x = float(
                    input("Enter the initial (scalar) value of \'x\': "))
                user_config["startingPoint"] = initial_x

                x0 = NewtonScalar(float(user_config["coefficients"][0]["a"]), float(user_config["coefficients"][0]["b"]),
                                  float(user_config["coefficients"][0]["c"]), float(user_config["coefficients"][0]["d"]),
                                  initial_x)

                print(f"Newton yields the minimum point: x0 = {x0}")

                x_found, y_found = gradient(a, b, c, d, initial_x)

                print(f"Gradient Descent yields the minimum point: x0 = {x_found}")

            elif StartingPointSelection == "A":
                # stanisz: Automatic starting point selection
                print("\'x\' will be drawn uniformly from [low, high].")

                low = float(
                    input("Enter the lower bound of the domain of \'x\' (low): "))
                high = float(
                    input("Enter the upper bound of the domain of \'x\' (high): "))

                initial_x = random.uniform(low, high)
                print("Chosen starting \'x\' is: ", initial_x)
                user_config["startingPoint"] = initial_x

                x0 = NewtonScalar(float(user_config["coefficients"][0]["a"]), float(user_config["coefficients"][0]["b"]),
                                  float(user_config["coefficients"][0]["c"]), float(user_config["coefficients"][0]["d"]),
                                  user_config["startingPoint"])

                print(f"Newton yields the minimum point: x0 = {x0}")

                x_found, y_found = gradient(float(user_config["coefficients"][0]["a"]), float(user_config["coefficients"][0]["b"]),
                                            float(user_config["coefficients"][0]["c"]), float(user_config["coefficients"][0]["d"]),
                                            user_config["startingPoint"])

                print(f"Gradient Descent yields the minimum point: x0 = {x_found}")

            else:
                print_bad_input_message()
                exit(1)

        elif FunctionSelection == "G":

            print("Enter the matrix \'A\': ")

            rows = int(input("Enter the number of rows: "))
            columns = int(input("Enter the number of columns: "))

            print("Enter the matrix elements one by one")

            A = []
            for i in range(rows):        
                a =[]
                for j in range(columns):
                    a.append(float(input("a" + str(i) + str(j) + " = ")))
                A.append(a)
            
            print(f"Enter the vector \'b\', {columns} numbers:")

            b = []
            for i in range(columns):
                b.append(float(input("b" + str(i) + " = ")))

            c = float(input("Enter the scalar value of coefficient \'c\': "))

            user_config["coefficients"].append({"A" : A, "b" : b, "c" : c})

            StartingPointSelection = input(
                "If you would like to enter initial value of \'x\' manually enter M. Enter \'A\' for automatic choice: ")
            
            if StartingPointSelection == "M":

                # stanisz: Manual starting point selection
                print(f"Enter the starting point \'x\', {columns} numbers: ")
                
                initial_x = []

                for i in range(columns):     
                    initial_x.append(float(input("x" + str(i) + " = ")))

                user_config["startingPoint"] = initial_x    

                x0 = NewtonMat(user_config["coefficients"][0]["A"], user_config["coefficients"][0]["b"],
                               user_config["coefficients"][0]["c"], user_config["startingPoint"])
                print(f"Newton yields the minimum point: x0 = {x0}")

                x_found, y_found, intermediate_x, intermediate_y = gradient2(user_config["coefficients"][0]["A"], user_config["coefficients"][0]["b"],
                               user_config["coefficients"][0]["c"], user_config["startingPoint"])

                print(f"Gradient Descent yields the minimum point: x0 = {x_found}")

            elif StartingPointSelection == "A":

                # stanisz: Automatic starting point selection
                print("\'x\' will be drawn uniformly from [low, high].")

                low = float(
                    input("Enter the lower bound of the domain of \'xi\' (low): "))
                high = float(
                    input("Enter the upper bound of the domain of \'xi\' (high): "))

                initial_x = np.ones([1, columns])
                initial_x = [x * random.uniform(low, high) for x in initial_x]
                print("Chosen starting \'x\' is: ", initial_x)
                user_config["startingPoint"] = initial_x

                x0 = NewtonMat(user_config["coefficients"][0]["A"], user_config["coefficients"][0]["b"],
                               user_config["coefficients"][0]["c"], user_config["startingPoint"])
                print(f"Newton yields the minimum point: x0 = {x0}")

                x_found, y_found, intermediate_x, intermediate_y = gradient2(user_config["coefficients"][0]["A"], user_config["coefficients"][0]["b"],
                               user_config["coefficients"][0]["c"], user_config["startingPoint"])

                print(f"Gradient Descent yields the minimum point: x0 = {x_found}")

        else:
            print_bad_input_message()

            exit(1)

    if 0:
        x_starting = -0.999999999
        a = 1
        b = 1
        c = -1
        d = 1

        print(f"Newton yields: x0 = {NewtonScalar(a, b, c, d, x_starting, 100)}")

        if 0: # TESTS

            stanisz_x = []
            stanisz_y = []

            for i in range(1, 200, 2):
                [x_found, y_found] = gradient(a, b, c, d, x_starting, i)
                if math.isnan(x_found) == False:
                    stanisz_x.append(i)
                    stanisz_y.append(abs(x_found - 1/3))

            plt.title("Error vs MaxIterations")
            plt.scatter(stanisz_x, stanisz_y)
            plt.show()

        x_found, y_found = gradient(a, b, c, d, x_starting)

        if math.isnan(x_found):
            print("Gradient Descent did NOT converge to a solution (-infinity)")
        else:
            print(f"Solution found by gradient descent = {x_found}")

    else: # plotting only for 3D spaces
        x_starting = [9.9, 9.9]
        c = 0
        B = [5, -2]
        A = [[1, 1], [0, 1]]
        
        #print(f"Newton yields: x0 = {NewtonMat(A, B, c, x_starting, 100)}")

        x_found, y_found, intermediate_x, intermediate_y = gradient2(
            A, B, c, x_starting, 400)

        #print(x_found, y_found)
        if 0 and len(x_starting) == 2:

            a1 = A[0][0]
            a2 = A[0][1]
            a3 = A[1][0]
            a4 = A[1][1]
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            # defining all 3 axes
            X = np.arange(-10, 10, 0.5)
            Y = np.arange(-10, 10, 0.5)
            X, Y = np.meshgrid(X, Y)
            Z = c + B[0]*X + B[1]*Y + a1*X**2+X*Y*(a2+a3)+a4*Y**2

            surf = ax.scatter([x[0] for x in intermediate_x],
                              [x[1] for x in intermediate_x],
                              intermediate_y,
                              color='red', s=50)

            surf = ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm)

            plt.show()

if __name__ == '__main__':
    main()
