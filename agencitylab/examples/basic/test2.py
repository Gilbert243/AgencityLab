import numpy as np
from agencitylab import pipeline

# ============================================================
# SIGNALS
# ============================================================

t = np.linspace(0, 10, 200)

signals = {
    "sinus": np.sin(t),
    "multi-scale": np.sin(t) + 0.3 * np.sin(5 * t),
    "bruit blanc": np.random.randn(200),
    "mixte": np.sin(t) + np.random.randn(200) * 0.5,
    "constant": np.ones_like(t),
}

# ============================================================
# TEST A_ref AUTOMATIQUE
# ============================================================

print("\n==============================")
print("TEST A_ref automatique (unit)")
print("==============================")

for i, (name, u) in enumerate(signals.items(), 1):
    print("\n" + "=" * 50)
    print(f"[TEST {i}] {name}")
    print("=" * 50)

    res = (
        pipeline()   # ✅ CORRECT
        .from_signal(u, xi=t)
        .set_unit("m")  # 🔥 A_ref = 1 m
        .run()
    )

    print("A_ref:", getattr(res, "A_ref", None))
    print("b_mean:", getattr(res, "b_mean", None))
    print("tau:", getattr(res, "tau", None))

    if getattr(res, "signature", None):
        print("signature slope:", res.signature["slope"])
        print("regime:", res.signature["regime"])


# ============================================================
# TEST OVERRIDE A_ref
# ============================================================

print("\n==============================")
print("TEST override A_ref")
print("==============================")

u = np.sin(t)

res = (
    pipeline()
    .from_signal(u, xi=t)
    .set_unit("rad")
    .set_reference_amplitude(0.1)
    .run()
)

print("A_ref (override):", res.A_ref)
print("b_mean:", res.b_mean)
print("tau:", res.tau)


# ============================================================
# TEST COMPARAISON
# ============================================================

print("\n==============================")
print("TEST comparaison A_ref")
print("==============================")

u = np.sin(t)

res1 = (
    pipeline()
    .from_signal(u, xi=t)
    .set_unit("rad")  # A_ref = 1
    .run()
)

res2 = (
    pipeline()
    .from_signal(u, xi=t)
    .set_unit("rad")
    .set_reference_amplitude(0.1)
    .run()
)

print("A_ref 1.0 → b_mean:", res1.b_mean)
print("A_ref 0.1 → b_mean:", res2.b_mean)


# ============================================================
# TEST CORE DIRECT
# ============================================================

print("\n==============================")
print("TEST core direct (sanity check)")
print("==============================")

from agencitylab.core.activation import compute_activation
from agencitylab.core.tau import estimate_tau

u = np.random.randn(200)

X = compute_activation(u, t)
tau = estimate_tau(X, axis=t)

print("tau (core):", tau)