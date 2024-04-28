import numpy as np
import time


def zlatyrez(f, a, b, x, ddd, tol):
    """
    Find the minimum of a function f using the Golden section method.

    Parameters
    ----------
    f : Callable
        Function to find the minimum of.
    a : float
        Left endpoint of the interval.
    b : float
        Right endpoint of the interval.
    x : float
        Point to optimize around.
    ddd : int
        Direction of the search.
    tol : float
        Tolerance of the method.

    Returns
    -------
    tuple[float, int]
        Tuple of the argument of minimum and number of iterations.
    """

    # Golden ratio
    gamma = 1 / 2 + np.sqrt(5) / 2

    # Initial values
    a0 = a
    b0 = b
    d0 = (b0 - a0) / gamma + a0
    c0 = a0 + b0 - d0

    # Iteration counter
    it = 0

    # Store the values of the interval and the function
    an = a0
    bn = b0
    cn = c0
    dn = d0
    fcn = f(x + cn * ddd)
    fdn = f(x + dn * ddd)

    while bn - an > tol:
        # Store the values of the interval and the function
        a = an
        b = bn
        c = cn
        d = dn
        fc = fcn
        fd = fdn

        if fc < fd:
            # Update the interval
            an = a
            bn = d
            dn = c
            cn = an + bn - dn

            # Update the function value
            fcn = f(x + cn * ddd)
            fdn = fc
        else:
            # Update the interval
            an = c
            bn = b
            cn = d
            dn = an + bn - cn

            # Update the function value
            fcn = fd
            fdn = f(x + dn * ddd)

        # Increment the iteration counter
        it += 1

    # Return the result
    t = (an + bn) / 2
    return t, it


def newton(f, df, ddf, x0, tolf=1e-6, tolg=1e-3, linesearch_tol=1e-3, maxit=100, verbose=False):
    """
    Newton's method for function minimization

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    df : function
        The gradient of the objective function.
    ddf : function
        The Hessian of the objective function.
    x0 : numpy.ndarray
        The initial guess for the minimum.
    tolf : float
        The tolerance for the stopping condition for function values.
    tolg : float
        The tolerance for the stopping condition for gradient values.
    linesearch_tol : float
        The tolerance for the line search method.
    maxit : int
        The maximum number of iterations.
    verbose : bool
        If True, print iteration information.

    Returns
    -------
    res : OptimizeResult from scipy.optimize._optimize
        The optimization result class with the following fields defined:
        x : ndarray
            The solution of the optimization.
        fun : float
            The value of the objective function at the solution.
        nit : int
            The number of iterations.
        message : str
            A string describing the cause of the termination.
    """

    # Start the timer
    start = time.perf_counter()

    x = x0
    fx = f(x)
    it = 0
    message = "Maximum number of iterations reached"

    for _ in range(maxit):
        # Gradient
        g = df(x)

        normg = np.linalg.norm(g)

        if normg < tolg:
            message = "Stopping condition for g is satisfied"
            break

        it += 1  # Iteration = Hessian evaluation + line search

        # Hessian
        H = ddf(x)
        # Newton's step
        h = -H.solve(g)

        # For stability of nolinear problems we search (-0.5, 2) x h
        a, nitf = zlatyrez(f, -0.5, 2, x, h, linesearch_tol)

        # Update x and function value
        x = x + a * h
        fxn, fx = fx, f(x)

        if verbose:
            print(f"it={it}, f={fx:.5f}, fstep = {fxn - fx:.5e}, ||g||={normg:.5f}, nitf={nitf}, a={a:.5e}")

        if np.abs(fx - fxn) < tolf:
            message = "Stopping condition for f is satisfied"
            break

    runtime = time.perf_counter() - start
    res = {"x": x, "fun": fx, "nit": it, "time": runtime, "message": message}

    return res
