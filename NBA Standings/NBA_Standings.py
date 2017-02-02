from bs4 import BeautifulSoup
import urllib.request
import os
import time

resp = urllib.request.urlopen('http://www.nba.com/standings/team_record_comparison/conferenceNew_Std_Cnf.html?ls=iref:nba:gnav')
doc = resp.read()
soup = BeautifulSoup(doc, 'html.parser')

f = open(os.path.expanduser("~/Desktop/standings.txt"),'w')
errorFile = open(os.path.expanduser("~/Desktop/errors.txt"),'w')

tableStats = soup.find("table", attrs={"class" : "genStatTable mainStandings"})

i = int('1')
flag = int ('0')
f.write("NBA REGULAR SEASON STANDINGS AS ON: ")
f.write(time.strftime("%c"))
f.write("\n\n")
print ("NBA Regular Season Standings as on :  " , time.strftime("%c"), "\n\n")
print ("Eastern Conference".center(10),"\n")
f.write("Eastern Conference\n\n")
for row in tableStats.find_all('tr')[2:]:
	print("\n")
	row_team = row.find_all("td")
	
	try:
		for stat in row_team:
			print("{0:>5} {1:>5} ".format(stat.text," "), end=" ")
			f.write("{0:^2} {1:^3} ".format(stat.text," "))
		if(i == 16 and flag == 0):
			i = int("0")
			flag = int('1')
			print("\n\n\n\n")
			print("Western Conference".center(10),"\n\n\n")
			f.write("Western Conference\n\n")
			
		i = i + 1
		f.write("\n")
	except Exception as e:   #In Case a none object gets returned
		pass



errorFile.close()
f.close()
