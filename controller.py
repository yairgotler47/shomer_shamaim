import serial
import time
import math
import numpy as np
SPD = 1000
TIKES = 800
DIR = 1
ser = serial.Serial('COM13', 115200, timeout=0.01)
ZERO_TOP = 20

def send_angles(angles_theta_phi: tuple):
    """
    gets angles in rad and send in steps and calc dir
    """
    theta = (angles_theta_phi[0]*50*2)/np.pi
    dir_t = 1 if theta<0 else 0
    phi = (angles_theta_phi[1]*50*2)/np.pi - ZERO_TOP
    phi = phi - ZERO_TOP
    dir_p = 1 if phi<0 else 0
    print(f"{dir_t} {theta} {dir_p} {phi}")
    #send_command(dir_t, abs(theta), dir_p, abs(phi))



def send_command(dir1, theta, dir2=0, phi=0, power_rifle = 0):
    """
    200 ticks = 360 deg
    if power_rifle  = 1  => change the state of the gun ON\OFF
    """

    print("hi")
    #phi = phi-20
    cmd = str(dir1) + ';' + str(theta).zfill(3) + ';' + str(dir2)+";"+str(phi).zfill(3) + ";" + str(power_rifle)
    print(cmd)
    #time.sleep(2)
    ser.write(cmd.encode())
    
def main():
    while(True):
        theta = int(input("enter theta: "))
        phi = -1
        while(phi<0 or phi>45):  
            phi = int(input("enter phi: "))
        power_rifle = int(input("enter one to make gun shoot: "))
        
        dir1 = 0
        if(theta<0):
            dir1 = 1
        dir2 = 1
        # if(phi<0):
        #     dir2 = 1
        
        send_command(dir1, abs(theta), dir2, abs(phi), power_rifle)
if __name__ == "__main__":
    main()
    
    


