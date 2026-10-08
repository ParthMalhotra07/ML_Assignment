from itertools import combinations_with_replacement
import numpy as np

def exponents(n_features, degree):
    powers = []
    for d in range(1, degree + 1):
        for inds in combinations_with_replacement(range(n_features), d):
            powers.append(np.bincount(inds, minlength=n_features))
    return np.asarray(powers)


def design(x, degree, basis="monomial"):
    powers = exponents(x.shape[1], degree)
    if basis == "monomial":
        uni = [x[:, j, None] ** np.arange(degree + 1) for j in range(x.shape[1])]
    else:
        uni = [np.polynomial.legendre.legvander(x[:, j], degree) for j in range(x.shape[1])]

    out = np.ones((len(x), len(powers)))
    for j in range(x.shape[1]):
        out *= uni[j][:, powers[:, j]]
    return out


def ridge_path(a, y, b, alphas):

    mu = a.mean(0)
    sd = a.std(0)
    sd[sd < 1e-12] = 1
    z = (a - mu) / sd
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    uy = u.T @ (y - y.mean())
    factors = s[:, None] / (s[:, None] ** 2 + np.asarray(alphas)[None, :])
    coefs = vt.T @ (factors * uy[:, None])
    return ((b - mu) / sd) @ coefs + y.mean()


def fit_model(x, y, degree, basis, alpha):
    a = design(x, degree, basis)
    mu, sd = a.mean(0), a.std(0)
    sd[sd < 1e-12] = 1
    u, s, vt = np.linalg.svd((a - mu) / sd, full_matrices=False)
    coef = vt.T @ ((s / (s*s + alpha)) * (u.T @ (y-y.mean())))
    return dict(degree=degree, basis=basis, alpha=alpha, mean=mu, scale=sd,
                coef=coef, intercept=float(y.mean()))


def predict(model, x):
    return ((design(x, int(model["degree"]), str(model["basis"])) - model["mean"])
            / model["scale"]) @ model["coef"] + model["intercept"]
