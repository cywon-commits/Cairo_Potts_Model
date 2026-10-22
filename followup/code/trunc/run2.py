import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys; sys.path.insert(0,ROOT+"/trunc")
from trunc import *
for L in [4,6]:
    analyze(f"truncated square (4.8.8)      L={L}", square(L))
for L in [3,4]:
    analyze(f"truncated dice                L={L}", dice(L))
    analyze(f"truncated honeycomb (3.12.12) L={L}", honeycomb(L))
