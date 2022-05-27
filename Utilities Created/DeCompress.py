
import os
import tarfile

filepath = "C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/Compressed"
files = os.listdir(filepath)
list_of_files = [filepath + "/" + x for x in files]

for element in list_of_files:
    file = tarfile.open(element)
    file.extractall("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/Pictures")
    file.close()
    print("completed " + element)
