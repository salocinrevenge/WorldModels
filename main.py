from motor import Motor

import os

# if "WAYLAND_DISPLAY" in os.environ:
#     del os.environ["WAYLAND_DISPLAY"]
# if "DISPLAY" not in os.environ:
#     os.environ["DISPLAY"] = ":0"
    
if __name__ == "__main__":
    motor = Motor()
    motor.run()
