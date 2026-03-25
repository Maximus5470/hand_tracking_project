import serial
import serial.tools.list_ports
import time
import tkinter as tk

# ===== SELECT PORT =====
ports = serial.tools.list_ports.comports()

if not ports:
    print("No serial ports found.")
    exit()

print("Available Ports:")
for i, port in enumerate(ports):
    print(f"{i}: {port.device}")

selection = int(input("Select port number: "))
PORT = ports[selection].device

arduino = serial.Serial(PORT, 9600)
time.sleep(2)

print(f"Connected to {PORT}")

# ===== GUI =====
root = tk.Tk()
root.title("Robot Arm Controller")
root.geometry("500x450")

NEUTRAL_CH3 = 104
angles = [0, 0, 0, 0, NEUTRAL_CH3]

def map_to_speed(value):
    value = float(value)
    centered = value - 135
    speed = int(NEUTRAL_CH3 + centered * 0.4)
    return max(70, min(140, speed))

def send_angles():
    message = ",".join(str(a) for a in angles) + "\n"
    arduino.write(message.encode())

def update_slider(index, value):
    if index == 4:
        angles[index] = map_to_speed(value)
    else:
        angles[index] = int(float(value))
    send_angles()

def stop_ch3(event):
    """
    Stop motor WITHOUT moving slider visually
    """
    angles[4] = NEUTRAL_CH3
    send_angles()

labels = ["Shoulder (Ch 0)",
          "Hand Speed (Ch 4)",
          "Elbow (Ch 2)",
          "Wrist (Ch 1)",
          "Base (Ch 3)"]

for i in range(5):
    slider = tk.Scale(root,
                      from_=0,
                      to=270,
                      orient="horizontal",
                      length=420,
                      label=labels[i],
                      command=lambda val, idx=i: update_slider(idx, val))

    if i == 4:
        slider.set(135)  # center start
        slider.bind("<ButtonRelease-1>", stop_ch3)
    else:
        slider.set(0)

    slider.pack(pady=6)

# Send initial position
send_angles()

def close_program():
    arduino.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", close_program)

root.mainloop()