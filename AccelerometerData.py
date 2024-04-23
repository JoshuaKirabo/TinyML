import smbus
import time
import csv
import os

# MPU6050 setup and addresses
DEVICE_ADDRESS = 0x68  # MPU6050 device address
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
SM_BUS_NUMBER = 1  # Might be 0 on older Raspberry Pi models

# Initialize the I2C bus
bus = smbus.SMBus(SM_BUS_NUMBER)

# Wake-up procedure for the MPU6050
bus.write_byte_data(DEVICE_ADDRESS, PWR_MGMT_1, 0)

# Function to read raw data from the accelerometer
def read_raw_data(addr):
    # Accelero data are 16-bit so need to be read in high and low parts
    high = bus.read_byte_data(DEVICE_ADDRESS, addr)
    low = bus.read_byte_data(DEVICE_ADDRESS, addr+1)

    # Combine high and low for final value
    value = ((high << 8) | low)
    
    # Correct signed value
    if(value > 32768):
        value = value - 65536
    return value

# Function to log data
def log_data(filename):
    # Check if file exists, create it with a header if it doesn't
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z"])
    
    # Open the file to append new data
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        while True:
            try:
                # Read acceleration data
                accel_x = read_raw_data(ACCEL_XOUT_H)
                accel_y = read_raw_data(ACCEL_XOUT_H + 2)
                accel_z = read_raw_data(ACCEL_XOUT_H + 4)
                
                # Log data to console
                print(f"Timestamp: {time.time()}, Accel X: {accel_x}, Accel Y: {accel_y}, Accel Z: {accel_z}")
                
                # Write the data to the csv file
                writer.writerow([time.time(), accel_x, accel_y, accel_z])
                
                # Wait a short time to simulate the sample rate you need
                time.sleep(0.1)  # Sleep for 100ms
            except KeyboardInterrupt:
                # Stop the loop with CTRL+C
                print("Data logging stopped")
                break

# Run the logging function
if __name__ == "__main__":
    log_data("accelerometer_data.csv")
