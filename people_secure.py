import pandas as pd
import csv
import numpy as np


file = open("people.csv")
csvreader = csv.reader(file)

csv = pd.read_csv('people.csv')

#csvreader = csv.reader(file)

csv.drop('First Name', inplace=True, axis=1)
csv.drop('Last Name', inplace=True, axis=1)

#print(csv)
csv.to_csv('release.csv')
#csv.close()



#TRIED DOING THE CODE WITHOUT THE USE OF THE DATAFRAME, HAD ISSUES
#PRINTING TO release.csv

#header = next(csvreader)[2:8]
#print(header)
#rows = []
#for row in csvreader:
    #rows.append(row[2:8])
#print(rows)
#file.close()

#np.savetxt('release.csv', rows, delimiter=',')
