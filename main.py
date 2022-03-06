from pyexpat.model import XML_CQUANT_NONE
import random
from shutil import move
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator

#stanisz: To have uniform messages for bad input
def print_bad_input_message():
    print("Bad input! Exiting...")

def gradient(a, b, c, d, x_starting, iterations_limit = 1000):
        f = np.polynomial.Polynomial([d, c, b, a])
        dx = 0.00000000001
        f_derivative = f.deriv()
        current_x = x_starting
        lowest_so_far = current_x
        for i in range(iterations_limit):
            if f(current_x) < f(lowest_so_far):
                lowest_so_far = current_x

            slope = f_derivative(current_x)
            if slope == 0: #stanisz: Floating precision issues!
                current_x += random.choice([-dx, dx])

            
            #i_factor = (1 - i / iterations_limit) / 10000
            #step = -0.05 * np.sign(slope) * slope**2 + i_factor 
            #step = -0.1 * np.sign(slope) * math.log(abs(slope) + 1)
            #step = -0.1 * np.sign(slope) * np.exp(slope)
            step = -0.1 * slope
            current_x += step
        return [lowest_so_far, f(lowest_so_far)]

def gradient2(A, B, c, x_starting, iterations_limit = 1000):        
        dx = 0.00000000001
        current_x = x_starting
        lowest_so_far = current_x
        value_lowest_so_far = sys.float_info.max
        intermediate_results_x = []
        intermediate_results_y = []
        for i in range(iterations_limit):
            value_current = c + np.inner(B, current_x) + np.inner(current_x, A.dot(current_x).tolist()[ 0 ])
            intermediate_results_x.append(current_x)
            intermediate_results_y.append(value_current)
            if value_current < value_lowest_so_far:
                lowest_so_far = current_x
                value_lowest_so_far = value_current

            if 1:
                slope = []
                for outer_i, x_i in enumerate(current_x):
                    move_vector = np.zeros(len(current_x))
                    move_vector[outer_i] = dx
                    xi_moved = current_x + move_vector
                    value_xi_moved = c + np.inner(B, xi_moved) + np.inner(xi_moved, A.dot(xi_moved).tolist()[ 0 ])
                    partial_xi = (value_xi_moved - value_current) / dx #stanisz: I hope!!
                    slope.append(partial_xi)
                slope = np.array(slope)
            else:
                t1 = np.matrix(current_x).transpose()
                test = np.array(current_x) * A
                #print(test)
                partial_xi = np.matrix(B).transpose() + 2 * test
                slope = np.array(partial_xi.transpose().tolist()[ 0 ])
                print(slope)
            step = -0.1 * slope
            current_x += step
            current_x = current_x.tolist()
        return [lowest_so_far, value_lowest_so_far, intermediate_results_x, intermediate_results_y]

def main():
    print("Welcome to gradient-py!")

    if 0:
        #stanisz: Can we simplify this logic?
        user_config = {}

        print("What is the stopping condition?")
        user_config["StoppingCondition"] = input("Enter I for max. iterations, V for value-to-reach, C for max. computation time:")
        
        if StoppingConditionSelection == "I":
            user_config["IterationsLimit"] = int(input("How many iterations is the computational limit?"))
        elif StoppingConditionSelection == "V":
            user_config["ValueToReach"] = float(input("What is the value-to-reach?"))
        elif StoppingConditionSelection == "C":
            user_config["MaxComputationTime"] = float(input("What is the maximum computation time (in seconds)?"))
        else:
            print_bad_input_message()
            exit(1)

        FunctionSelection = input("To compute F(x) enter F, to compute G(x) enter G:")
        if FunctionSelection == "F":
            #stanisz: Computing F
            a = float(input("Enter the scalar value of coefficient \'a\':"))
            b = float(input("Enter the scalar value of coefficient \'b\':"))
            c = float(input("Enter the scalar value of coefficient \'c\':"))
            d = float(input("Enter the scalar value of coefficient \'d\':"))
            StartingPointSelection = input("If you would like to enter initial value of \'x\' manually enter M. Enter \'A\' for automatic choice:")
            if StartingPointSelection == "M":
                #stanisz: Manual starting point selection
                initial_x = float(input("Enter the initial (scalar) value of \'x\':"))
            elif StartingPointSelection == "A":
                #stanisz: Automatic starting point selection
                print("\'x\' will be drawn uniformly from [low, high].")
                low = float(input("Enter the lower bound of the domain of \'x\' (low):"))
                high = float(input("Enter the upper bound of the domain of \'x\' (high):"))
                initial_x = random.uniform(low, high)
            else:
                print_bad_input_message()
                exit(1)
        elif FGselection == "G":
            #stanisz: Computing G
            print("Currently not supported! Exiting...")
        else:
            print_bad_input_message()
            exit(1)
    #stanim: Config disabled, main starts here:

    if 0: 
        x_starting = -0.999999999
        a = 1
        b = 1
        c = -1
        d = 1

        if 1:
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

        [x_found, y_found] = gradient(a, b, c, d, x_starting)
        if math.isnan(x_found):
            print("Gradient Descent did NOT converge to a solution (-infinity)")
        else:
            print(f"Solution found by gradient descent = {x_found}")
    else:
        x_starting = [7.5, 7.5]
        c = 0
        B = [5, -2]
        A_raw = [[1, 1], [0, 1]]
        A = np.matrix(A_raw)
        [x_found, y_found, intermediate_x, intermediate_y] = gradient2(A, B, c, x_starting, 40)
        print(x_found, y_found)
        if 1 and len(x_starting) == 2:
            a1 = A_raw[0][0]
            a2 = A_raw[0][1]
            a3 = A_raw[1][0]
            a4 = A_raw[1][1]
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # defining all 3 axes
            X = np.arange(-10, 10, 0.5)
            Y = np.arange(-10, 10, 0.5)
            X, Y = np.meshgrid(X, Y)
            Z = c + B[0]*X + B[1]*Y + a1*X**2+X*Y*(a2+a3)+a4*Y**2
            #print(intermediate_x, "DUPA", intermediate_y)
            
            surf = ax.scatter([x[0] for x in intermediate_x], [x[1] for x in intermediate_x], intermediate_y, color='red', s=50)
            
            surf = ax.plot_wireframe(X, Y, Z, cmap=cm.coolwarm)
            # Customize the z axis.
            #ax.set_zlim(-1.01, 1.01)
            #ax.zaxis.set_major_locator(LinearLocator(10))
            # A StrMethodFormatter is used automatically
            #ax.zaxis.set_major_formatter('{x:.02f}')

            # Add a color bar which maps values to colors.
            #fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.show()

if __name__ == '__main__':    
    main()