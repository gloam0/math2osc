#!/usr/bin/env python3
import argparse
import numpy as np
from numpy.fft import rfft


def fourier_sin_series(func, T=2*np.pi, N=16384, K=10):
    """
    Approximate a periodic f(t) by a sine series using FFT

    Samples f(t) uniformly on [0,T) and computes the FFT, returning coeffs
    for the approximation:

    f(t) = a0 + SUM{k=1..K}(Ak * sin(k*w0*t + phik))

    where:
        - w0 = 2π / T   fundamental *angular* frequency (rad / unit t)
        - k*w0          kth harmonic angular frequency
        - Ak            sine amplitude of the kth harmonic
        - phik          sine phase (radians)
    ---
    func : a (callable) function that can be evaluated on an array of
           times. Ex: `func = lambda t: np.sin(t + np.sin(2*t))`
    T    : period of `func`
    N    : number of uniformly spaced samples over one period, prefer
           powers of two for efficiency
    K    : number of harmonics in the sine series to return, K <= N/2 (nyquist)
    ---
    a0    : DC offset (mean value over period)
    w0    : fundamental angular frequency (2pi/T)
    terms : list of (k, Ak, phik) for k=1..K, representing Ak*sin(k*w0*t + phik)
    """
    if K > N // 2:
        raise ValueError(f"K must be <= N/2. Got K={K}, N={N}.")

    t = np.linspace(0, T, N, endpoint=False)  # samples
    f_t = func(t)      # f(t) evaluated for each sample

    X = rfft(f_t) / N  # fft coeffs over f_t
    a0 = X[0].real     # DC offset

    w0 = 2*np.pi / T   # fundamental angular frequency
    terms = []         # output
    for k in range(1, K+1):
        ck = X[k]                     # complex coefficient for exp(+i k w0 t)
        Ak_cos = 2*np.abs(ck)         # amplitude
        phi_cos = np.angle(ck)        # phase

        # convert cos to sin: cos(t) = sin(t + pi/2)
        Ak_sin = Ak_cos
        phi_sin = phi_cos + np.pi/2

        # wrap phase
        phi_sin = (phi_sin + np.pi) % (2*np.pi) - np.pi

        terms.append((k, Ak_sin, phi_sin))
    return a0, w0, terms


def fourier_sin_series_to_callable(a0, w0, terms, tolerance=1e-12):
    """
    Build a callable approximation from a0, w0, terms (see fourier_sin_series)
    ---
    tol    : drop terms with |amplitude| < tol
    ---
    approx : approx(t) evaluates:
               a0 + SUM{k=1..K}(Ak * sin(k*w0*t + phik))
             works for scalar t or numpy array t
    """
    kept = [(k, Ak, phik) for (k, Ak, phik) in terms if abs(Ak) >= tolerance]

    def approx(t):
        t = np.asarray(t)
        y = a0 + np.zeros_like(t, dtype=float)
        for k, Ak, phik in kept:
            y += Ak * np.sin(k * w0 * t + phik)
        # supports scalar t or numpy array t
        return float(y) if y.shape == () else y

    return approx


def str_to_func(func_str):
    np_funcs_map = {
        "pi": np.pi,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
    }
    allowed = {"np": np}
    code = compile(args.func, "<args.func>", "eval")
    func = lambda t: eval(code, {"__builtins__": {}}, {"t": t, **SAFE_FUNCS, "np": np})
    return func

# a0, w0, terms = fourier_sin_series(lambda t: np.sin(t+np.sin(2*t)), T=2*np.pi, N=16384, K=10)
# print(a0)
# print(w0)
# print(terms)
# f_t = fourier_sin_series_to_callable(a0, w0, terms)
# print(f_t(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("func")
    
    args = parser.parse_args()

    allowed = {"np": np}
    code = compile(args.func, "<args.func>", "eval")
    f = lambda t: eval(code, {"__builtins__": {}}, {"t": t, **allowed})

    a0, w0, terms = fourier_sin_series(f, T=2*np.pi, N=16384, K=10)
    print(a0)
    print(w0)
    print(terms)
    f_t = fourier_sin_series_to_callable(a0, w0, terms)
    print(f_t(10))
