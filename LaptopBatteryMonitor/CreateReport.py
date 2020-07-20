from time import strftime
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

### load the file
with open("BatteryMonitorData.txt", "r") as file:
    lines = file.readlines()
    
### categories
categories = lines[0].split(",")
for i,cat in enumerate(categories):
	categories[i] = cat.strip()
Ncat = len(categories)

### data as list of strings
data = lines[1:]
Nrow = len(data)

def trp(l, n):
    return l[:n] + [""]*(n-len(l))
    
### split each string into a list
for i,row in enumerate(data):
	data[i] = row.split(",")
	data[i] = trp(data[i], Ncat)

# strip each string from whitespace and newline	
for i,row in enumerate(data):
	for j,elt in enumerate(row):
		data[i][j] = elt.strip()

# turn into nparray
npdata = np.array(data)
npdata = np.transpose(npdata)

### named individual columns
date = npdata[0]
cyclecount = npdata[1]
fullycharged = npdata[2]
chargeremaining = npdata[3]
fullchargecapacity = npdata[4]
voltage = npdata[5]

### change data types of columns
date = date.astype("O")
for i,d in enumerate(date):
	date[i] = datetime.strptime(d,"%d-%m-%Y %H:%M")
cyclecount = cyclecount.astype(np.float)
chargeremaining = chargeremaining.astype(np.float)
fullchargecapacity = fullchargecapacity.astype(np.float)
voltage = voltage.astype(np.float)

### count number of dates in past week / day
Ninweek = 0
Ninday = 0
now = datetime.now()
for d in date:
	dif = now-d
	dif = dif.days
	if(dif<7): Ninweek = Ninweek+1
	if(dif<1): Ninday = Ninday+1

### create figure
fig = plt.figure(constrained_layout=False, figsize=(16, 8))
fig.subplots_adjust(wspace=0.15, hspace=0.1)

gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:1])
ax2 = fig.add_subplot(gs[1, 0:1],sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0:1],sharex=ax1)

ax4 = fig.add_subplot(gs[0, 1:2]						)
ax5 = fig.add_subplot(gs[1, 1:2],sharex=ax4	,sharey=ax2)
ax6 = fig.add_subplot(gs[2, 1:2],sharex=ax4	,sharey=ax3)

ax7 = fig.add_subplot(gs[0, 2:3]						)
ax8 = fig.add_subplot(gs[1, 2:3],sharex=ax7	,sharey=ax2)
ax9 = fig.add_subplot(gs[2, 2:3],sharex=ax7	,sharey=ax3)

ax1.set_title("All time")
ax1.plot(date, cyclecount, color='blue')
ax2.plot(date, fullchargecapacity, color='red')
ax3.plot(date, voltage, color='green')
ax1.set_ylabel('Cycle Count')
ax2.set_ylabel('Full Charge Capacity (mAh)')
ax3.set_ylabel('Voltage (mV)')

# hist = 5*24
hist = Ninweek
ax4.set_title("Past week")
ax4.plot(date[-hist:-1], cyclecount[-hist:-1], color='blue')
ax5.plot(date[-hist:-1], fullchargecapacity[-hist:-1], color='red')
ax6.plot(date[-hist:-1], voltage[-hist:-1], color='green')

# hist = 1*24
hist = Ninday
ax7.set_title("Past day")
ax7.plot(date[-hist:-1], cyclecount[-hist:-1], color='blue')
ax8.plot(date[-hist:-1], fullchargecapacity[-hist:-1], color='red')
ax9.plot(date[-hist:-1], voltage[-hist:-1], color='green')

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
ax7.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))

ax1.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")
ax2.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")
ax3.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")
ax4.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")
ax5.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")
ax6.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")
ax7.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")
ax8.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")
ax9.fmt_xdata = mdates.DateFormatter("%d-%m-%Y %H:%M")

fig.autofmt_xdate(bottom=0.2, rotation=60)
fig.suptitle("Battery Statistics")

# store the most recent figure in BatteryStatistics.png
plt.savefig("BatteryReport.png")
