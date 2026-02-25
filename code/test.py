import numpy as np

data = np.load('/data/gradovcic/output/20250226_Chengwei_Take8/obj_pose.npy')
print(f"Shape: {data[0].shape}")
print(f"Dtype: {data.dtype}")
print(f"Data:\n{data[0]}")