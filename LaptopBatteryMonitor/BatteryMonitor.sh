#!/bin/bash

DATE=$( date +"%d-%m-%Y %H:%M" )
CC=$( system_profiler SPPowerDataType | grep 'Cycle Count' | cut -f 2 -d':' )
FC=$( system_profiler SPPowerDataType | grep 'Fully Charged' | cut -f 2 -d':' )
CR=$( system_profiler SPPowerDataType | grep 'Charge Remaining' | cut -f 2 -d':' )
FCP=$( system_profiler SPPowerDataType | grep 'Full Charge Capacity' | cut -f 2 -d':' )
V=$( system_profiler SPPowerDataType | grep 'Voltage' | cut -f 2 -d':' )

echo $DATE,$CC,$FC,$CR,$FCP,$V >> BatteryMonitorData.txt

python3.7 CreateReport.py