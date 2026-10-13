# """Show a live camera feed until the preview is closed."""

# import argparse
# import time

# import cv2


# def main():
# 	parser = argparse.ArgumentParser(description="Keep a camera preview running.")
# 	parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
# 	camera_index = parser.parse_args().camera

# 	camera = cv2.VideoCapture(camera_index)
# 	if not camera.isOpened():
# 		camera.release()
# 		raise RuntimeError(f"Unable to open camera {camera_index}")

# 	window = "Camera — press Q or Esc to close"
# 	try:
# 		while True:
# 			success, frame = camera.read()
# 			if not success:
# 				# Keep the device open and retry after a brief transient failure.
# 				time.sleep(0.1)
# 				continue

# 			cv2.imshow(window, frame)
# 			key = cv2.waitKey(1) & 0xFF
# 			if key in (ord("q"), 27):
# 				break
# 			try:
# 				if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
# 					break
# 			except cv2.error:
# 				break
# 	except KeyboardInterrupt:
# 		pass
# 	finally:
# 		camera.release()
# 		cv2.destroyAllWindows()


# if __name__ == "__main__":
# 	main()

import cv2
import time
import argparse

camera_index = 0
camera = cv2.VideoCapture(camera_index)
if not camera.isOpened():
    camera.release()
    raise RuntimeError(f"Unable to open camera {camera_index}")

try:
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue

        cv2.imshow("Camera — press Q or Esc to close", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        try:
            if cv2.getWindowProperty("Camera — press Q or Esc to close", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
finally:
    camera.release()
    cv2.destroyAllWindows()