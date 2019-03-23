import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#################################################################
features = ['salary', 'experience', 'gender']

# probably tech field
ave_sal_male = 100000
ave_sal_female = 80000

# us workers
ave_sal_male = 49192
ave_sal_female = 39988

# i think this is right
std_dev_sal = 10000

# uneven amount of data
num_samples_female = 10000
num_samples_male = 40000

# I do not have anything that says 20 years is average, just a guess
ave_experience = 10
std_dev_experience = 3
###################################################################

# create a dummy dataset
# salary, gender
salary_female = []
salary_male = []

stds = [std_dev_sal, std_dev_experience]
corr = 0.8

covs = [[stds[0]**2, stds[0]*stds[1]*corr],
        [stds[0]*stds[1]*corr, stds[1]**2]]

sal_female_arr = np.random.multivariate_normal([ave_sal_female, ave_experience], covs, num_samples_female).T
sal_male_arr = np.random.multivariate_normal([ave_sal_male, ave_experience], covs, num_samples_male).T

female_mins = np.min(sal_female_arr, axis=1)
male_mins = np.min(sal_male_arr, axis=1)

########shift#########################################
if female_mins[0] < 0:
        sal_female_arr[0] += abs(female_mins[0])

if female_mins[1] < 0:
        sal_female_arr[1] += abs(female_mins[1])

if male_mins[0] < 0:
        sal_male_arr[0] += abs(male_mins[0])

if male_mins[1] < 0:
        sal_male_arr[1] += abs(male_mins[1])
######################################################

plt.scatter(sal_female_arr[0],sal_female_arr[1])
plt.show()
plt.scatter(sal_male_arr[0],sal_male_arr[1])
plt.show()

sal_female_arr = sal_female_arr.T
sal_male_arr = sal_male_arr.T 

# add gender columns
sal_female_arr = np.append(sal_female_arr, np.zeros((10000,1)),axis=1)
sal_male_arr = np.append(sal_male_arr, np.ones((40000,1)),axis=1)

sal_arr = np.vstack((sal_female_arr,sal_male_arr))
print(sal_arr.shape)
np.random.shuffle(sal_arr)
df = pd.DataFrame(data=sal_arr, columns=features)

df.to_csv('salary_data.csv', sep=',')
