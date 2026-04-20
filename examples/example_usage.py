

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


file_path = "patch2.png"

img = Image.open(file_path).convert("L")
img_np = np.array(img)


# Compute all LQP variants
lqp1 = compute_RIU4_LQP(img_np, mode='LQP1', R=1, P=8, T1=2, T2=7)

# lqp2 = compute_RIU4_LQP(img_np, mode='LQP2', R=1, P=8, T1=5, T2=10)
# lqp3 = compute_RIU4_LQP(img_np, mode='LQP3', R=1, P=8, T1=5, T2=10)
# lqp4 = compute_RIU4_LQP(img_np, mode='LQP4', R=1, P=8, T1=5, T2=10)


# Show one result
plt.imshow(lqp1, cmap='gray')
plt.title("RIU4-LQP1")
plt.colorbar()
plt.show()