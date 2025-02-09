from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util)
from cupyx.profiler import benchmark
import nvtx

bench = Benchmark("compute")
frmwrk = generate_framework("cupy", save_strict=False, load_strict=False)
numpy = generate_framework("numpy")
# lcount = LineCount(bench, frmwrk, numpy)
# lcount.count()

test = Test(bench, frmwrk, numpy)
test.run("S", False, 2, None)
