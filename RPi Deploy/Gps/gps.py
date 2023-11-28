import serial
import pynmea2
import json

""" 
GPS Precision by decimal places

decimal |
places  |   degrees     |   distance
0	        1.0	            111 km
1	        0.1	            11.1 km
2	        0.01	        1.11 km
3	        0.001	        111 m
4	        0.0001	        11.1 m
5	        0.00001	        1.11 m      (We use this for this application)
6	        0.000001	    0.111 m
7	        0.0000001	    1.11 cm
8	        0.00000001	    1.11 mm
"""

precision = 5
counter = 0
lat  = 0
lng = 0

jsonFile = {
    "lat": 0,
    "lng": 0,
}

while True:
    port='/dev/serial0'
    ser=serial.Serial(port, baudrate=9600, timeout=0.5)
    dataout = pynmea2.NMEAStreamReader()
    newdata=ser.readline().decode()
    if newdata[0:6] == '$GPRMC':
        # Receiving data every second
        newmsg = pynmea2.parse(newdata)
        # Average of 5 measurements
        if counter ==5:
            jsonFile["lat"] = round(lat/5,precision)
            jsonFile["lng"] = round(lng/5,precision)
            lat = jsonFile["lat"] 
            lng = jsonFile["lng"]
            gps = 'Latitude= ' + str(lat) + ' and Longitude= ' + str(lng)
            #print(gps)
            with open("opencvDNN/Gps/gps.json", "w") as outfile:
                json.dump(jsonFile, outfile) 
            lat= 0
            lng = 0
            counter = 0
        elif (newmsg.latitude==0.0 or newmsg.longitude ==0.0):
            # Avoiding failed values
            continue
        else:
            lat +=newmsg.latitude
            lng +=newmsg.longitude
            counter+=1