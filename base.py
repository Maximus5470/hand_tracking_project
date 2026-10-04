import serial
import serial.tools.list_ports
import time
import tkinter as tk

# =============================
# Select Serial Port
# =============================

ports = list(serial.tools.list_ports.comports())

if not ports:
    print("No serial ports found.")
    exit()

print("Available Ports:")
for i, port in enumerate(ports):
    print(f"{i}: {port.device}")

selection = int(input("Select port number: "))
PORT = ports[selection].device

try:
    arduino = serial.Serial(PORT, 9600, timeout=1)
    time.sleep(2)
    print(f"Connected to {PORT}")
except Exception as e:
    print("Failed to connect:", e)
    exit()

# =============================
# Servo Angles
# =============================

angles = [90, 90, 90, 90, 90]

labels = [
    "Base (CH0)",
    "Shoulder (CH4)",
    "Elbow (CH2)",
    "Wrist (CH1)",
    "Gripper (CH3)"
]

# =============================
# Send Servo Data
# =============================

def send_angles():
    message = ",".join(map(str, angles)) + "\n"
    arduino.write(message.encode())
    print(message.strip())

# =============================
# Slider Callback
# =============================

def slider_changed(index, value):
    angles[index] = int(float(value))
    send_angles()

# =============================
# GUI
# =============================

root = tk.Tk()
root.title("5-DOF Robot Arm Controller")
root.geometry("550x500")

for i in range(5):
    slider = tk.Scale(
        root,
        from_=0,
        to=180,
        orient=tk.HORIZONTAL,
        length=450,
        label=labels[i],
        command=lambda value, idx=i: slider_changed(idx, value)
    )

    slider.set(90)
    slider.pack(pady=8)

# Send initial positions
send_angles()

# =============================
# Close Program
# =============================

def on_close():
    if arduino.is_open:
        arduino.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()