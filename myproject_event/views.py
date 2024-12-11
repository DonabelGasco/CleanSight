import os
import base64
import uuid
import cv2
import numpy as np
import json
import torch
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import logging
from django.shortcuts import render
from .models import Detection
from datetime import datetime
from .models import CapturedImage
from django.utils.timezone import now
from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages



# Initialize logger for debugging
logger = logging.getLogger(__name__)



# Initialize the YOLO model
model = YOLO("myproject_event/models/bestest.pt")

UPLOAD_FOLDER = os.path.join(settings.BASE_DIR, "myproject_event/static/uploads")
OUTPUT_FOLDER = os.path.join(settings.BASE_DIR, "myproject_event/static/outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

streaming = False


def home(request):
    return render(request, 'home.html')

def live(request):
    global streaming
    streaming = True  # Start streaming
    return render(request, 'live.html')

def stop_live(request):
    global streaming
    streaming = False  # Stop streaming
    return render(request, 'home.html')

def real_time_detections(request):
    detections = Detection.objects.all().order_by('-timestamp')[:10]  # Fetch the latest 10 detections
    return render(request, 'real_time_detections.html', {'detections': detections})
def generate_frames():
    global streaming
    cap = cv2.VideoCapture(0)  # Open the webcam

    while streaming:
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLO detection on the frame
        results = model.predict(source=frame, save=False)
        annotated_frame = results[0].plot()

        # Get the detections
        detections = results[0].boxes
        for box in detections:
            label = model.names[int(box.cls[0])]  # Class label
            confidence = float(box.conf[0])  # Confidence score

            if label == "waste" and confidence > 0.5:  # Assuming "waste" is one of the detected classes
                # Capture and save the image
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bbox = f"{int(box.xyxy[0][0])},{int(box.xyxy[0][1])},{int(box.xyxy[0][2])},{int(box.xyxy[0][3])}"

                # Save the image
                image_filename = f"{uuid.uuid4().hex}.jpg"
                image_filepath = os.path.join(UPLOAD_FOLDER, image_filename)
                cv2.imwrite(image_filepath, frame)  # Save the frame as an image

                # Save the detection to the database
                detection = Detection.objects.create(
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    image=image_filename  # Store the image filename
                )
                detection.save()

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame in a byte-stream format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    # Yield an empty frame to close the stream
    yield (b'')

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def capture(request):
    return render(request, 'capture.html')

@csrf_exempt
def capture_live(request):
    if request.method == 'POST':
        try:
            # Parse Base64 Image
            data = json.loads(request.body).get("image")
            if not data:
                return JsonResponse({"error": "No image data received"}, status=400)

            image_data = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Save Image
            input_filename = f"{uuid.uuid4().hex}_live_input.jpg"
            input_filepath = f"/static/uploads/{input_filename}"
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, input_filename), img)

            # YOLO Detection
            results = model.predict(source=img, save=False)
            annotated_img = results[0].plot()

            output_filename = f"{uuid.uuid4().hex}_live_output.jpg"
            output_filepath = f"/static/outputs/{output_filename}"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, output_filename), annotated_img)

            # Save to Database
            response_data = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                label = model.names[cls]

                response_data.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

            CapturedImage.objects.create(
             image_path=input_filepath,
              annotated_path=output_filepath,
               detections=response_data,
               timestamp=now(),
              source='capture'  # Indicate the source is "Capture"
)

            return JsonResponse({"output": output_filepath, "detections": response_data})

        except Exception as e:
            logger.error(f"Error in capture_live: {e}")
            return JsonResponse({"error": "Server error"}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)

def captured_images_list(request):
    captures = CapturedImage.objects.all().order_by('-timestamp')
    return render(request, 'captured_images_list.html', {'captures': captures})

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Save uploaded image
            file = request.FILES['image']
            fs = FileSystemStorage(location=UPLOAD_FOLDER)
            filename = fs.save(file.name, file)
            input_path = os.path.join(UPLOAD_FOLDER, filename)

            # Load the image
            img = cv2.imread(input_path)
            if img is None:
                return JsonResponse({"error": "Failed to read the uploaded image"}, status=400)

            # Perform YOLO detection
            results = model.predict(source=img, save=False)
            annotated_img = results[0].plot()

            # Save annotated image
            output_filename = f"{uuid.uuid4().hex}_output.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, annotated_img)

            # Save to the database
            detections_data = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                label = model.names[cls]

                detections_data.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

                captured_image = CapturedImage.objects.create(
                image_path=f"/static/uploads/{filename}",
                annotated_path=f"/static/outputs/{output_filename}",
                detections=detections_data,
                timestamp=now(),
                source='upload' 
)


            return JsonResponse({
                "output": f"/static/outputs/{output_filename}",
                "detections": detections_data,
                "image_id": captured_image.id
            })

        except Exception as e:
            return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)

    return render(request, 'upload_image.html')
def processed_images_list(request):
    images = CapturedImage.objects.all().order_by('-timestamp')
    return render(request, 'processed_images_list.html', {'images': images})

def delete_image(request, image_id):
    if request.method == "POST":
        image = get_object_or_404(CapturedImage, id=image_id)
        
        # Optionally, delete files from the server
        if os.path.exists(os.path.join(settings.BASE_DIR, image.image_path.strip("/"))):
            os.remove(os.path.join(settings.BASE_DIR, image.image_path.strip("/")))
        if os.path.exists(os.path.join(settings.BASE_DIR, image.annotated_path.strip("/"))):
            os.remove(os.path.join(settings.BASE_DIR, image.annotated_path.strip("/")))

        # Delete the image record from the database
        image.delete()

        # Add a success message
        messages.success(request, "Image deleted successfully.")
    return redirect("processed_images_list")  # Redirect back to the list page



def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_FOLDER)
        filename = fs.save(video_file.name, video_file)
        input_video_path = fs.url(filename)

        # Define output video path
        output_filename = f"{uuid.uuid4().hex}_output.mp4"
        output_video_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Process the 
        # video
        try:
            cap = cv2.VideoCapture(input_video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Default FPS if unavailable
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO detection
                results = model.predict(source=frame, save=False)
                annotated_frame = results[0].plot()

                # Write the annotated frame
                out.write(annotated_frame)

            cap.release()
            out.release()

            # Verify file exists
            if os.path.exists(output_video_path):
                return JsonResponse({"output": f"/static/outputs/{output_filename}"})
            else:
                return JsonResponse({"error": "Processed video not found"}, status=500)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, 'upload_video.html')

def serve_video(request, filename):
    return HttpResponse(open(os.path.join(OUTPUT_FOLDER, filename), 'rb').read(), content_type='video/mp4')
