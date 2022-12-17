# Citeseer
# P, N:  903 1093

# p=0.9 with Mixup
# TP:  826, TN:  1087.0, FP:  6,  FN:  77.0

# p=0.7 with Mixup
# TP:  658, TN:  1090.0, FP:  3,  FN:  245.0, F1: 0.84

# p=0.6 with Mixup
# TP:  555, TN:  1092.0, FP:  1,  FN:  348.0, F1: 0.76

# p=0.5 with Mixup
# TP:  462, TN:  1092.0, FP:  1,  FN:  441.0, F1: 0.67


# p=0.1 with Mixup
# TP:  112, TN:  1092.0, FP:  1,  FN:  791.0, F1: 0.22

"""
  # ===== test ===== #
  p=0.9
    Precision:  tensor(0.7054, device='cuda:0')   Recall:  tensor(0.8667, device='cuda:0')  F1:  tensor(0.7778, device='cuda:0')
  
  p=0.6
    Precision:  tensor(0.6801, device='cuda:0')   Recall:  tensor(0.7492, device='cuda:0')  F1:  tensor(0.7130, device='cuda:0')
  
  p=0.1
    Precision:  tensor(0.6117, device='cuda:0')   Recall:  tensor(0.3651, device='cuda:0')  F1:  tensor(0.4573, device='cuda:0')

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# TP, FP,  FN, TN
# array = [[826, 6], [77, 1087]]  # train, p=0.9
# array = [[658, 3], [245, 1090]]
# array = [[555, 1], [348, 1092]]
# array = [[462, 1], [441,1092]]
# array = [[112, 1], [791, 1092]]
# array = [[273, 114], [42, 237]]  # test, p=0.9
# array = [[252, 110], [63, 241]]  # test, p=0.6
# array = [[115, 73], [200, 278]]  # test, p=0.1


# array = [[238, 2], [188, 1568]]
array = [[382, 26], [44, 1544]]
cm = pd.DataFrame(array)

fig, ax = plt.subplots()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, annot_kws={"size": 15})
plt.xlabel('actual values', fontsize=15)
plt.ylabel('predicted values', fontsize=15)
plt.xticks([0.5, 1.5], ['P', 'N'], fontsize=15)
plt.yticks([0.5, 1.5], ['P', 'N'], fontsize=15)

# moving x axis to the top of a plot
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

plt.savefig('confusion_matrix_05_Citeseer_PU-GCN-S.png')
