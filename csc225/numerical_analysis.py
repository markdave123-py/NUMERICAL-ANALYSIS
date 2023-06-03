import ast
import sympy as sym
import numpy as np
import math

def start():
    print("Welcome to rootfinder and ODE solver. What would you like to do?")
    print("1.-- Find a root of an equation")
    print("2.-- Solve a first order Ordinary Differential Equation")
    choice_num = int(input("Input choice number:"))
    print()
    if choice_num == 1:
        findEquRoot()
    elif choice_num == 2:
        solveODE()
    else:
        print("Sorry. Your choice is not a valid choice.")

def findEquRoot():
    begin = "yes"
    while begin == "yes":
        print("Input the function you want to find the root of below. Make sure 'x' is the variable you use")
        print("For example . x**5 - 2*x")
        s = "lambda x:"
        s += input("Input function:")
        f = eval(compile(ast.parse(s, mode='eval'), filename='', mode='eval'))
        try:
            f(1) #evaluting for errors
        except:
            raise Exception("Invalid function! Did you use the variable x?")
    
        print("Which method do you want to use? Input the corresponding value")
        print("Bisection: 1")
        print("Newton Raphson: 2")
        print("Secant: 3")
    
        method = int(input("Input method value:"))
        
        if method == 1:
            a = float(input("Input x1:"))
            b = float(input("Input x2:"))
            tol = float(input("Input tolerance:"))
            print(myBisection(f, a, b, tol))
        elif method == 2:
            x = float(input("Input x:"))
            tol = float(input("Input tolerance:"))
            x1 = sym.Symbol('x')
            fprime = differentiate(f)
            print(mynewtonRaphson(f, fprime, x, tol))
        elif method == 3:
            x0 = float(input("Input x0:"))
            x1 = float(input("Input x1:"))
            tol = float(input("Input tolerance:"))
            itr = int(input("Input the number of iterations:"))
            print(mySecant(f, x0, x1, tol, itr))
        else:
            print("That is an invalid selection")
        
        begin = input("Do you want to solve for another root? Yes or no:").lower()
    return

def solveODE():
    end = "no"
    while end == "no":
        print("Using variables x and y input the first order ODE you want to solve:")
        s = "lambda x, y:"
        s += input("Input function:")
        f = eval(compile(ast.parse(s, mode='eval'), filename='', mode='eval'))
    
        try:
            f(0, 0)
        except:
            raise Exception("Make sure you use only variables x and y!!!")
    
        print("What method do you want to use to solve it?")
        print("1. Euler Method")
        print("2. Picard Method")
        print("3. Taylor Method")
    
        method = int(input("Enter method number:"))
        if method == 1:
            x = float(input("Initial value of x:"))
            y = float(input("Initial value of y:"))
            dx = float(input("Step size (example ->. 0.1):"))
            itr = int(input("Number of iterations:"))
            print(round(euler(x, y, f, dx, itr), 4))
        elif method == 2:
            x = float(input("Initial value of x:"))
            y = float(input("Initial value of y:"))
            x0 = float(input("Particular value of x:"))
            itr = int(input("Number of iterations:"))
            print(round(picard(f, x, y, x0, itr), 3))
        elif method == 3:
            x = float(input("Initial value of x:"))
            y = float(input("Initial value of y:"))
            x0 = float(input("Particular value of x:"))
            dx = float(input("Step size (example ->. 0.1):"))
            order = int(input("Order of Taylor series?"))
            print(round(taylor(f, x, y, x0, order, dx), 3))
        else:
            print("Invalid selection")
        
        end = input("Do you want to solve for another ODE?Yes or no:").lower()
    return
        
    
def myBisection(f, a, b, tol):    
    if  f(a)*f(b) > 0:
        print(a, b)
        raise Exception(f"The chosen points {a} and {b} cannot bound to a root of the equation")
    
    m = (a + b) / 2
    
    if np.abs(f(m)) < tol:
        return m    
    elif np.sign(f(a)) == np.sign(f(m)):
        return myBisection(f, m, b, tol)  #recursively
    
    elif np.sign(f(b)) == np.sign(f(m)):
        return myBisection(f, a, m, tol)  #recursively

def differentiate(f):    
    x = sym.Symbol('x')
    f1 = sym.sympify(f(x))
    fprime = f1.diff(x)
    fprime = sym.lambdify([x], fprime)
    return fprime

    
def mynewtonRaphson(f, fprime, x, tol):
    if np.abs(f(x)) < tol:
        return x
    try:
        x = x - f(x)/fprime(x)
    except ZeroDivisionError:
        print("Error , you can't divide by Zero try another value for x")
        
    return mynewtonRaphson(f, fprime, x, tol)

def mySecant(f, x0, x1, tol, itr):
    err = np.abs(x1 - x0)
    x2 = 0
    
    if err > tol:
        for i in range(itr):
            try:
                x2 = x1 - f(x1) * (x0 - x1) / (f(x0) - f(x1))
            except ZeroDivisionError:
                print("Error,you cannot divide by zero")
                print(f'x0: {x0}\nx1: {x1}\nf(x0): {f(x0)}\nf(x1): {f(x1)}')
                break
            x0 = x1
            x1 = x2
            err = np.abs(x1 - x0)
            if err < tol: #the root is found 
                break
    
    return x1


def euler(x, y, f, dx, itr):
    for i in range(itr):
        try:
            print('x = %.5f | y = %.5f | dy/dx = %.5f' %(x, y, f(x, y)))
            y = y + (dx * f(x, y))
            x += dx
        except:
            print("Unknown error occured")
            break
    return round(y, 4)

def picard(d, x0, y0, x, itr):
    x_sym = sym.Symbol('x')
    y_sym = sym.Symbol('y')
    
    f = sym.sympify(d(x_sym, y_sym))
    y = y0
    
    for i in range(itr):
        integral = sym.integrate(f.subs({y_sym:y}), (x_sym, x0, x_sym))
        y = y0 + integral
        
    return y.subs({x_sym:x})

def taylor(d, x0, y0, x, order = 4, step = .2):
    x_sym = sym.Symbol('x')
    y_sym = sym.Symbol('y')
    
    f = sym.sympify(d(x_sym, y_sym))
    
    while x0 < x: 
        f0 = f
        y = y0
        e = str(y)
        e1 = str(f0)
        for j in range(order):
            expr = (step**(j+1)) * f0.subs({x_sym:x0, y_sym:y0}) / (math.factorial((j+1)))
            e += " '+' " + str(expr)
            y += expr
            f0 = f0.diff(x_sym) + f0.diff(y_sym) * f.subs({x_sym:x0, y_sym:y0})
            
            e1 += " + " + str(f0)
        y0 = y
        x0 += step
        print(e1)
        print("=",e)
        print(f'y({x0}) = {round(y,4)}')
        print('\n\n\n')
    return y

if __name__ == "__main__":
    stop = 'no'
    while stop == 'no':
        start()
        print("Would you like to end the program? Or start again?")
        stop = int(input("Enter yes to stop, no to continue")).lower()