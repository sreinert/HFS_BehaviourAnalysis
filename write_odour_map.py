odour_settings = {
    "Dev1/port0/line3": 4,
    "Dev1/port0/line4": 1,
    "Dev1/port0/line7": 13,
    "Dev1/port0/line2": 14,
    "Dev1/port0/line5": 18,
    "Dev1/port1/line6": 19,
    "Dev1/port1/line7": 17,
    "Dev1/port1/line4": 6,
    "Dev1/port2/line5": 12,
    "Dev1/port2/line4": 11,
}

modd_settings = {
    "Dev1/port0/line3": 1.3,
    "Dev1/port0/line4": 1.2,
    "Dev1/port0/line7": 1.4,
    "Dev1/port0/line2": 1.5,
    "Dev1/port0/line5": 1.6,
    "Dev1/port1/line6": 2.5,
    "Dev1/port1/line7": 2.4,
    "Dev1/port1/line4": 2.6,
    "Dev1/port2/line5": 2.2,
    "Dev1/port2/line4": 2.3,
}

with open("odour_settings.txt", "w") as file:
    for key, value in odour_settings.items():
        file.write(f"{key}: {value}\n")

with open("modd_settings.txt", "w") as file:
    for key, value in modd_settings.items():
        file.write(f"{key}: {value}\n")

