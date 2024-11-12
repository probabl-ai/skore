"""
.. _example_historization_items:

==================================
Tracking items using their history
==================================

This example illustrates [...].
"""

# %%
import subprocess
import time
import random

import matplotlib.pyplot as plt
import skore

# %%
random.seed(0)

# %%
# Creating and loading the skore project
# ======================================

# %%

# remove the skore project if it already exists
subprocess.run("rm -rf my_project_hist.skore".split())

# create the skore project
subprocess.run("python3 -m skore create my_project_hist".split())


# %%
my_project_hist = skore.load("my_project_hist.skore")

# %%
my_project_hist.put("my_int", 3)
time.sleep(0.1)
my_project_hist.put("my_int", 4)
time.sleep(0.1)
my_project_hist.put("my_int", 5)

# %%
my_list = my_project_hist.get_item_versions("my_int")
my_list

# %%
print(my_list[0])
print(my_list[0].primitive)
print(my_list[0].created_at)
print(type(my_list[0].created_at))

# %%
print(my_list[1])
print(my_list[1].primitive)
print(my_list[1].created_at)
print(type(my_list[1].created_at))

# %%
list_values = [elem.primitive for elem in my_list]
list_create = [elem.created_at for elem in my_list]
list_update = [elem.updated_at for elem in my_list]

print(list_create)
print(list_update)


# %%
plt.plot(list_update, list_values)
plt.xticks(rotation=45)
plt.show()

# %%
