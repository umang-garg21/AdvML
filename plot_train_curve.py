import matplotlib.pyplot as plt
import numpy as np

epoch_lst = [i for i in range(300)]
f1 = "/home/teamteam/AdvML/HW2/accuracy_adv_lst.csv"
f2 =  "/home/teamteam/AdvML/HW2/accuracy_clean_lst.csv"
accuracy_adv_lst= np.loadtxt(f1)
accuracy_clean_lst= np.loadtxt(f2)

plt.figure()
# plt.plot(epoch_lst, loss_lst, label='learning curve')
plt.plot(epoch_lst, accuracy_clean_lst, label='Clean accuracy')
plt.plot(epoch_lst, accuracy_adv_lst, label='Robust Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Accuracies vs epoch_Advanced_aT_After200epochs.png')
