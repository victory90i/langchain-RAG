# Classroom Monitoring & Alarm System

## Setup

1. Clone the repo.
2. Create and activate a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS / Linux
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Download the two MediaPipe model files into the repo root:

   - `pose_landmarker_full.task`
   - `face_landmarker.task`

   Both are available from the MediaPipe models page:
   https://ai.google.dev/edge/mediapipe/solutions/vision

## Run

From the folder containing this project:

```
python -m classroom.main             # webcam / default source 0
python -m classroom.main video.mp4   # video file
```

Or directly:

```
python classroom\main.py
python classroom\main.py sample-video.mp4
```
