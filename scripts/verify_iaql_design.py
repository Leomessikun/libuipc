"""Independent numerical check of the proposed IAQL Bellman derivative."""
import json
import numpy as np

rng = np.random.default_rng(19)
K, n, d, nz = 3, 2, 2, 5
mass = np.diag([1.2, 0.8])
H = [mass + np.diag([0.8 + k * 0.1, 1.4]) for k in range(K)]
A = np.zeros((K*n, K*n))
P = np.zeros((K*n, d))
offset = rng.normal(size=K*n) * 0.1
for k in range(K):
    sl = slice(k*n, (k+1)*n)
    A[sl, sl] = H[k]
    if k > 0:
        A[sl, (k-1)*n:k*n] = -2*mass
    if k > 1:
        A[sl, (k-2)*n:(k-1)*n] = mass
    P[sl] = ((k+1)/K) * np.diag([0.006, 0.012])
D = np.linalg.solve(A, P)
Z = np.zeros((nz, K*n))
Z[:n, -n:] = np.eye(n)
Z[n:2*n, -n:] = np.eye(n) / 0.1
Z[n:2*n, -2*n:-n] = -np.eye(n) / 0.1
Zu = np.zeros((nz, d))
Zu[-1] = [0.02, -0.03]  # direct control/history dependence
Wmu = rng.normal(size=(d, nz)) * 0.1
Wls = rng.normal(size=(d, nz)) * 0.03
eps = np.array([0.7, -0.4])
R = np.diag([0.6, 0.9])
U = np.diag([0.3, 0.5])
C = rng.normal(size=(nz, d)) * 0.1
cost_x = np.diag(np.linspace(0.02, 0.15, K*n))
u = np.array([0.23, -0.31])

def continuation(z, version):
    ls = Wls @ z - 0.6
    std = np.exp(ls)
    pre = Wmu @ z + std * eps
    ap = np.tanh(pre)
    pre_z = Wmu + (std*eps)[:, None]*Wls
    ap_z = (1-ap**2)[:, None]*pre_z
    Vmat = np.diag(np.linspace(0.2, 0.8, nz)) * version
    q = -0.5*z@Vmat@z + z@C@ap - 0.5*ap@R@ap
    q_z = -Vmat@z + C@ap + ap_z.T@(C.T@z-R@ap)
    logpi = -0.5*np.sum(eps**2 + np.log(2*np.pi)) - ls.sum() - np.log1p(-ap**2).sum()
    logpi_z = -Wls.sum(axis=0) + 2*pre_z.T@ap
    alpha = 0.17
    return q-alpha*logpi, q_z-alpha*logpi_z

def evaluate(u, mask, bound, version=1.0):
    x = np.linalg.solve(A, offset + P@u)
    z = Z@x+Zu@u
    w, wz = continuation(z, version)
    yraw = -0.5*x@cost_x@x - 0.5*u@U@u + 0.93*mask*w
    b = -cost_x@x + 0.93*mask*Z.T@wz
    direct = -U@u + 0.93*mask*Zu.T@wz
    lam = np.linalg.solve(A.T, b)
    active = float(abs(yraw) < bound)
    adj = active*(direct+P.T@lam)
    tangent = active*(direct+D.T@b)
    return np.clip(yraw, -bound, bound), adj, tangent, b, direct

records = []
for mask in [0, 1]:
    for bound in [10.0, 0.001]:
        for version in [1.0, 1.7]:
            val, adj, tan, b, direct = evaluate(u, mask, bound, version)
            for step in [1e-4, 1e-5, 1e-6]:
                eye = np.eye(d)*step
                fd = np.array([(evaluate(u+e, mask, bound, version)[0]-evaluate(u-e, mask, bound, version)[0])/(2*step) for e in eye])
                error = float(np.linalg.norm(adj-fd)/max(1e-8, np.linalg.norm(fd)))
                assert error < 1e-5, (mask, bound, version, step, error)
                assert np.allclose(adj, tan, atol=1e-12)
                records.append(error)

_, g1, _, b, direct = evaluate(u, 1, 10)
_, g2, tangent2, _, _ = evaluate(u, 1, 10, 1.7)
stale_gap = float(np.linalg.norm(g1-g2))
assert stale_gap > 1e-4
assert np.allclose(g2, tangent2, atol=1e-12)
last_only = direct + P[-n:].T@np.linalg.solve(H[-1].T, b[-n:])
last_only_error = float(np.linalg.norm(last_only-g1)/np.linalg.norm(g1))
assert last_only_error > 0.05
lam = np.linalg.solve(A.T, b)
perturbed = lam + rng.normal(size=K*n)*0.02
res = b - A.T@perturbed
g_perturbed = direct+P.T@perturbed
assert np.allclose(g1-g_perturbed, P.T@np.linalg.solve(A.T, res), atol=1e-12)
assert np.linalg.norm(g1-g_perturbed) <= np.linalg.norm(D, 2)*np.linalg.norm(res)+1e-12
print(json.dumps({"finite_difference_cases": len(records), "max_relative_error": max(records), "adjoint_tangent_agreement_atol": 1e-12, "last_frame_only_relative_error": last_only_error, "old_vs_refreshed_target_gradient_gap": stale_gap, "residual_error_identity_and_bound": "passed", "scope": "synthetic NumPy equations, not production IPC/SAC"}, indent=2))
