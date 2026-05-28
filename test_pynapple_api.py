import pynapple as nap
import numpy as np
import inspect

# Test 1: compute_tuning_curves return shape
ep = nap.IntervalSet(start=0, end=100)
feature = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.arange(0, 100, 0.1) % (2*np.pi), time_support=ep)
ts_group = nap.TsGroup({0: nap.Ts(np.arange(0, 100, 0.5)), 1: nap.Ts(np.arange(0, 100, 1.0))}, time_support=ep)
tc = nap.compute_tuning_curves(data=ts_group, features=feature, bins=10)
print("compute_tuning_curves xarray shape:", tc.shape)
print("dims:", tc.dims)
tc_pd = tc.to_pandas()
print("to_pandas shape:", tc_pd.shape)
print("to_pandas index (first 3):", tc_pd.index[:3].tolist())
print("to_pandas columns (first 3):", list(tc_pd.columns)[:3])
print()

# Test 2: TsGroup indexing
print("ts_group.keys():", list(ts_group.keys()))
try:
    result = ts_group[0]
    print("ts_group[0] type:", type(result).__name__)
except Exception as e:
    print("ts_group[0] error:", type(e).__name__, repr(e))

# Test 3: TsGroup.__getitem__ source snippet
src = inspect.getsource(nap.TsGroup.__getitem__)
print("\nTsGroup.__getitem__ source (first 1000 chars):")
print(src[:1000])
