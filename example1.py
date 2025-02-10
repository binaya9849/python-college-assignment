def f(x):
    return x**3 + 2*x**2 - 3.5

def bisection_method(a, b, tol=0.1):
    # Ensure that the function changes signs at the endpoints
    if f(a) * f(b) >= 0:
        print("The function does not change sign on the given interval.")
        return None

    # Start the bisection method
    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0  # Midpoint
        if f(c) == 0:  # Found the exact root
            return c
        elif f(c) * f(a) < 0:  # Root is in the left half
            b = c
        else:  # Root is in the right half
            a = c
    return (a + b) / 2.0  # Return the midpoint as the root

# Interval (1, 2)
a = 1
b = 2

root = bisection_method(a, b)
if root is not None:
    print(f"Root: {round(root, 1)}")
