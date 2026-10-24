# iso_tables.py: the two 1-D lookups of the recommended exact-safe stop rule (fit on dev cycles of both traces),
# printed on a grid, so the rule can be implemented as a table in common/speculative.cpp.
#   S_hat(d) = prod_{i<d} iso_q(q_i) * iso_p1(p1_d); do not add drafted token d when S_hat(d) < theta.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import heads as HD
iq = HD.isotonic_q(); ip = HD.isotonic_q(key="d_p1")
g = np.r_[0.0, 0.02, 0.05, np.arange(0.1, 0.951, 0.05), 0.98, 0.995, 1.0]
print("x      iso_q(x) [sampled token's q -> a]   iso_p1(x) [step top-1 p -> a]")
for x in g:
    print(f"{x:5.3f}  {iq.predict([x])[0]:.3f}                              {ip.predict([x])[0]:.3f}")
