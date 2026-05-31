import cv2
import dlib
import time
import os
import threading
import numpy as np
import csv
import io
import subprocess
import sys
import signal
import queue
import RPi.GPIO as GPIO

from PIL import Image
from imgbeddings import imgbeddings
from flask import Flask, Response, jsonify
from flask_cors import CORS
from flask import send_from_directory
from scipy.spatial.distance import euclidean
import tflite_runtime.interpreter as tflite
from MCP3008 import MCP3008
from RPLCD.i2c import CharLCD


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUZZER, GPIO.OUT, initial=GPIO.LOW)
BUZZER = 18   # GPIO Pin 18
last_lcd_time = 0
last_lcd_message = None
LCD_HOLD = 2
lcd = None
try: #Check whether LCD is detected
    lcd = CharLCD('PCF8574', 0x27)
    print("[INFO] LCD initialized at address 0x27")
except Exception as e:
    print(f"[WARN] LCD not detected or failed to initialize: {e}")
# ----------------------------
# Flask Setup
# ----------------------------
app = Flask(__name__)
CORS(app)   # Allow all origins
latest_image = None
stop_auth = False
auth_running = False
drowsy_running = False

auth_status = "Idle"
face_status = "Normal"
eye_status = "Normal"
yawn_status = "Normal"
head_status = "Normal"
alcohol_status = "No Alcohol Detected"
auth_score = 0.0
# ----------------------------
# SETTINGS
# ----------------------------
MATCH_THRESHOLD = 0.90 #Threshold for face recognition
DETECTION_INTERVAL = 2  # seconds between detection runs
EMBEDDINGS_FILE = "db_face.npy" #Driver database embedding file, consist of 12 driver face embeddings
buzzer_queue = queue.Queue()
# ----------------------------
# GLOBAL STATE
# ----------------------------
stop_face_authorize = False
success_count = 0
fail_count = 0

print("[INFO] Loading detector, shape predictor, and embeddings model...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
ibed = imgbeddings()

if not os.path.exists(EMBEDDINGS_FILE):
    print("[ERROR] No embeddings file found! Run generate_embedding.py first.")
    db_faces = []
else:
    db_faces = np.load(EMBEDDINGS_FILE)
print(f"[INFO] Loaded {len(db_faces)} embeddings.")

# ----------------------------------------------------------Drowsiness Detection Setup-------------------------------------------------------
# Set calibration threshold
ear_values = []
ntr_values = []

collecting = False
threshold_set = False
last_buzz_time = 0

# Eye-closed threshold and time
EAR_THRESHOLD = None
drowsy_counter = 0
drowsy_start = None

# Yawning threshold and time
MAR_THRESHOLD = 0.80
yawn_start = None
yawn_counter = 0

# Head bend down time
NOSE_DOWN_THRESHOLD = None
down_start = None
down_counter = 0
ntr = 0

# Load drowsiness detection model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Alcohol detection
adc = MCP3008()
ALCOHOL_THRESHOLD = 0.4

# History records
event_records = []

# Camera initialization
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
#-------------------------------------------------------------------------------------------------------------------------------------------
def lcd_display(lines):
    global last_lcd_time, last_lcd_message
    if lcd is None:
        print("[WARN] No LCD available, skipping display")
        return
    try:
        lcd.clear()
        for i, text in enumerate(lines):
            lcd.write_string(text)
            if i == 0 and len(lines) > 1:
                lcd.crlf()
        last_lcd_time = time.time()
        last_lcd_message = lines
    except Exception as e:
        print(f"[ERROR] LCD display failed: {e}")
# -----------------------------------
# FUNCTIONS FOR FACE AUTHORIZATION
# -----------------------------------
def safe_crop(img, box):
    h, w = img.shape[:2]
    x1, y1 = max(0, box.left()), max(0, box.top())
    x2, y2 = min(w, box.right()), min(h, box.bottom())
    if x2 <= x1 or y2 <= y1:
        return None, None
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def align_face(img, shape):
    # Align face using eye coordinates
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return aligned

def face_authorization():
    global latest_image, auth_status, auth_score, auth_running
    global stop_auth, success_count, fail_count, EAR_THRESHOLD, NOSE_DOWN_THRESHOLD
    threshold_set = False
    EAR_THRESHOLD = None
    NOSE_DOWN_THRESHOLD = None
    ear_values = []
    ntr_values = []
    lcd_display(["Authenticating..."]) 
    cap = cam
    # warm-up: grab a couple of frames so camera is ready
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    last_detection_time = 0
    auth_status = "Running"
    auth_running = True

    while not stop_auth:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Failed to capture frame")
            continue  
        now = time.time()
        if now - last_detection_time > DETECTION_INTERVAL:
            detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
            if not detections:
                auth_status = "No face detected"

            else:
                f = detections[0]
                shape =predictor(frame, f)

                x, y, w, h = f.left(), f.top(), f.width(), f.height()
                face_crop = frame[y:y+h, x:x+w]

                if face_crop is not None and face_crop.size > 0:
                    pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).resize((160, 160))
                    query_emb = ibed.to_embeddings(pil_face)[0]

                    # cosine similarity
                    score = max(
                        np.dot(query_emb, db_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(db_emb))
                        for db_emb in db_faces
                    )
                    
                    auth_score = float(score)
        
                    if score > MATCH_THRESHOLD:
                        auth_status = "success"
                        success_count += 1
                                            
                    else:
                        auth_status = "fail"
                        fail_count += 1

                    _, buffer = cv2.imencode(".jpg", frame)
                    latest_image = buffer.tobytes()
                    
                    # collect EAR
                    
                    if not threshold_set:
                        left_eye = [(shape.part(n).x, shape.part(n).y) for n in range(42, 48)]
                        right_eye = [(shape.part(n).x, shape.part(n).y) for n in range(36, 42)]
                        avgEAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                        ear_values.append(avgEAR)

                        # collect NTR
                        ntr = nose_tip_eye_vertical_ratio(shape)
                        ntr_values.append(ntr)

                        # after enough samples, compute thresholds
                        if len(ear_values) >= 3 and len(ntr_values) >= 3:
                            EAR_open = sum(ear_values) / len(ear_values)
                            EAR_THRESHOLD = EAR_open * 0.88

                            NTR_base = sum(ntr_values) / len(ntr_values)
                            margin = 0.88
                            NOSE_DOWN_THRESHOLD = NTR_base * margin

                            print(f"[CALIB] EAR={EAR_THRESHOLD:.2f}, NLR={NOSE_DOWN_THRESHOLD:.2f}")
                            threshold_set = True
                    
                        else:
                            print("Not enough frame")
                                   
                    else:
                        # fallback full frame
                        _, buffer = cv2.imencode(".jpg", frame)
                        latest_image = buffer.tobytes()

            last_detection_time = now

        time.sleep(0.1)
        
        if(success_count >= 3):
            lcd_display(["Authenticate", "Success :)"]) 
            success_count = 0
            time.sleep(1)

        
        elif (fail_count >= 3):
            lcd_display(["Authenticate", "Failed :("]) 
            fail_count = 0
            time.sleep(1)

    auth_running = False
    auth_status = "Stopped"
      
# -----------------------------------
# FUNCTIONS FOR DROWSINESS DETECTION
# -----------------------------------
def buzzer_worker():
    while True:
        action, duration = buzzer_queue.get()
        GPIO.output(BUZZER, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(BUZZER, GPIO.LOW)
        buzzer_queue.task_done()

# Start worker once at program start
threading.Thread(target=buzzer_worker, daemon=True).start()

def buzz_short(duration=0.3):
    buzzer_queue.put(("short", duration))

def buzz_long(duration=0.8):
    buzzer_queue.put(("long", duration))
    
def eye_aspect_ratio(eye_points):
    A = euclidean(eye_points[1], eye_points[5])  # vertical 1
    B = euclidean(eye_points[2], eye_points[4])  # vertical 2
    C = euclidean(eye_points[0], eye_points[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear
    
def predict_drowsiness(face_img, input_size=(32, 32)):

    # Preprocess image
    img = cv2.resize(face_img, input_size)
    img = img / 255.0                  # normalize 
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = int(np.argmax(output_data))
    return predicted_class, output_data

def eye_distance(landmarks):
    left_eye_outer = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye_outer = np.array([landmarks.part(45).x, landmarks.part(45).y])
    return np.linalg.norm(right_eye_outer - left_eye_outer)
    
def nose_tip_eye_vertical_ratio(landmarks):
    tip_y = landmarks.part(33).y
    chin_y = landmarks.part(8).y
    ed = eye_distance(landmarks)
    if ed <= 0:
        return 0.0
    return (chin_y - tip_y) / ed

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51-59
    B = np.linalg.norm(mouth[4] - mouth[8])   # 53-57
    C = np.linalg.norm(mouth[0] - mouth[6])   # 49-55
    return (A + B) / (2.0 * C)

def check_alcohol():
    value = adc.read(channel=0)
    voltage = round(value / 1023 * 3.3, 3)
    if voltage > ALCOHOL_THRESHOLD:
        return True, voltage
    return False, voltage

def drowsiness_detector():
    global threshold_set, ear_values, ntr_values, yawn_counter, down_counter, drowsy_counter
    global down_start, drowsy_running, eye_status, yawn_status, yawn_start, head_status, alcohol_status, drowsy_start, face_status
    global ntr, collecting, last_buzz_time, last_lcd_message, stop_auth
    drowsy_running = True
    cap = cam
    
    while True:	
        ret, frame = cap.read()
        if not ret:
            continue
        clean_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        alcohol_detected, voltage = check_alcohol()
        averageEAR = 0.0
        mar = 0.0
        ntr = 0.0
            
        if alcohol_detected:
            cv2.putText(frame, "ALCOHOL DETECTED!", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if time.time() - last_buzz_time > 3:
                buzz_short()
                lcd_display(["Alcohol Detected!"])          
                alcohol_status = "Alcohol Detected"
                last_buzz_time = time.time()
            
        else:
            alcohol_status = "Normal"
    
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            drowsy_start = None  # reset timer
            down_start = None
            yawn_start = None
            face_status = "No Face Detected"
            eye_status = "Unavailable"
            yawn_status = "Unavailable"
            head_status = "Unavailable"

        for face in faces:
            landmarks = predictor(gray, face)
            face_status = "Normal"
            face_count = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Eye close detection~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
            # Draw eye landmarks
            for n in range(36, 48):  # Both eyes
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                # ~ cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Extract left and right eye coordinates
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        
            # Calculate EAR
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            averageEAR = (leftEAR + rightEAR) / 2.0
        
       
            # Check EAR threshold
            if averageEAR <= EAR_THRESHOLD:
                if drowsy_start is None:
                    drowsy_start = time.time()  # start timing
                else:
                    eye_elapsed = time.time() - drowsy_start
                    if eye_elapsed >= 0.8 and not collecting:
                        collecting = True
                        window_start = time.time()
                        
                    if collecting:
                        pred_class, probs = predict_drowsiness(frame)
                        confidence = probs[0][0]
                        print(f"The confidence of this prediction is: {confidence*100:.3f}%")
                            
                        if pred_class == 0 and confidence >= 0.885:
                            drowsy_counter += 1
                            
                            if time.time() - window_start >= 0.5:
                                if drowsy_counter >= 5:
                                    
                                    cv2.putText(frame, "DROWSY!!!", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                                    face_img = clean_frame[y:y+h, x:x+w]
                                    # Save frame once
                                    _, buffer = cv2.imencode('.jpg', clean_frame)
                                    latest_image = buffer.tobytes()
                                    
                                    buzz_long()
                                    lcd_display(["Warning!!!", "Drowsy Detected"])  
                                    eye_status = "Drowsiness Detected"
                        
                                    prediction = "Drowsy"                                
                            
                                    filename = f"event_{time.time()}.jpg"
                                    filepath = os.path.join("events", filename)
                                    cv2.imwrite(filepath, clean_frame)

                                    # push record
                                    event_records.append({
                                        "type": "Drowsiness",
                                        "reason": "Eye closed for long time",
                                        "image": filename,
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "prediction": prediction,
                                        "confidence": f"{confidence*100:.2f}%"
                                        })
                                
                                else:
                                    eye_status = "Normal"
                                    
                                collecting = False
                                drowsy_start = None
                        else:
                            drowsy_counter = 0
                            collecting = False
                            drowsy_start = None
                                
            else:
                drowsy_start = None  # reset if eyes open
                eye_status = "Normal"
                collecting = False
                drowsy_counter = 0
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Yawning Detection~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
            # Mouth coordinate
            coords = np.array([[p.x, p.y] for p in landmarks.parts()])
            mouth = coords[48:68]
            mar = mouth_aspect_ratio(mouth)
        
            if mar >= MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = time.time()
                else:
                    mouth_elapsed = time.time() - yawn_start
                    if mouth_elapsed >= 2:
                        buzz_short()
                        lcd_display(["Warning!!!", "Drowsy Detected"]) 
                        
                        _, buffer = cv2.imencode('.jpg', clean_frame)
                        latest_image = buffer.tobytes()
                        cv2.putText(frame, "YAWNING!", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        yawn_counter += 1
                        yawn_start = None
                        yawn_status = "Yawning Detected"
                        
                        filename = f"event_{time.time()}.jpg"
                        filepath = os.path.join("events", filename)
                        cv2.imwrite(filepath, clean_frame)

                        # push record
                        event_records.append({
                            "type": "Drowsiness",
                            "reason": "Yawning",
                            "image": filename,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "prediction": "None",
                            "confidence": "None"
                            })    
            else:
                yawn_start = None  # reset
                yawn_status = "Normal"

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Head Bend Down Detection~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculate NLR
            ntr = nose_tip_eye_vertical_ratio(landmarks)        
   
            if ntr <= NOSE_DOWN_THRESHOLD:
                if down_start is None:
                    down_start = time.time()
        
                else:
                    nose_elapsed = time.time() - down_start
                    if nose_elapsed >= 2.5:
                    
                        buzz_long()
                        lcd_display(["Warning!!!", "Drowsy Detected"]) 
                        
                        _, buffer = cv2.imencode('.jpg', clean_frame)
                        latest_image = buffer.tobytes()
                        cv2.putText(frame, "HEAD DOWN!", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        down_counter += 1
                        down_start = None
                        head_status = "Head Bend Down Detected"
                        filename = f"event_{time.time()}.jpg"
                        filepath = os.path.join("events", filename)
                        cv2.imwrite(filepath, clean_frame)

                        # push record
                        event_records.append({
                            "type": "Drowsiness",
                            "reason": "Unnatural head position",
                            "image": filename,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "prediction": "None",
                            "confidence": "None"
                            })    
            else:

                down_start = None
                head_status = "Normal"
                    
        cv2.putText(frame, f"EAR: {averageEAR:.2f}", (7, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (7, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"NLR: {ntr:.2f}", (7, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Alcohol: {voltage:.2f}", (7, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        if last_lcd_message and (time.time() - last_lcd_time > LCD_HOLD):
            try:
                lcd.clear()
                last_lcd_message = None
            except Exception as e:
                print(f"[ERROR] LCD display failed: {e}")
        cv2.imshow("Drowsiness Detection", frame)
        cv2.waitKey(50)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # ESC key
            break

    drowsy_running = False
    cv2.destroyAllWindows()

def generate_frames():
    while True:
        success, frame = cam.read()
        if not success:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/shutdown_pi", methods=["POST"])
def shutdown_pi():
    GPIO.output(BUZZER, GPIO.LOW)
    try:
        if lcd is not None:
            lcd.clear()
    except Exception as e:
        print(f"[WARN] Could not clear LCD: {e}")
    GPIO.cleanup()
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)

    return {"status": "Pi program terminated"}, 200

@app.route("/auth/start", methods=["POST"])
def start_auth():
    global auth_running, stop_auth, success_count, fail_count
    if auth_running:
        return jsonify({"status": "already running"})
    success_count = 0
    fail_count = 0
    stop_auth = False
    threading.Thread(target=face_authorization, daemon=True).start()

    auth_running = True
    return jsonify({"status": "started"})

@app.route("/auth/stop", methods=["POST"])
def stop_auth_route():
    global stop_auth, auth_running
    stop_auth = False
    auth_running = False
    return jsonify({"status": "stopped"})

@app.route("/auth/status", methods=["GET"])
def status():
    return jsonify({"status": auth_status, "score": round(auth_score, 3)})

@app.route("/auth/image", methods=["GET"])
def get_image():
    global latest_image
    if latest_image is not None:
        return Response(latest_image, mimetype="image/jpeg")

    # return blank placeholder image
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/jpeg")
    
@app.route("/start/drowsy", methods=["POST"])
def start_drowsiness():
    global drowsy_running, stop_auth, auth_running
    stop_auth = True
    t0 = time.time()
    while auth_running and time.time() - t0 < 3.0:
        time.sleep(0.05)

    if drowsy_running:
        return jsonify({"status": "already running"})

    threading.Thread(target=drowsiness_detector, daemon=True).start()
    return jsonify({"status": "drowsiness started"})

@app.route("/events")
def get_events():
    return jsonify(event_records)

@app.route("/events/images/<filename>")
def get_event_image(filename):
    return send_from_directory("events", filename)

@app.route("/drowsy/face")
def get_face_status():
    return jsonify({"face_status": face_status})

@app.route("/drowsy/eye")
def get_eye_status():
    return jsonify({"eye_status": eye_status})

@app.route("/drowsy/yawn")
def get_yawn_status():
    return jsonify({"yawn_status": yawn_status})

@app.route("/drowsy/down")
def get_head_status():
    return jsonify({"head_status": head_status})

@app.route("/drowsy/alcohol")
def get_alcohol_status():
    return jsonify({"alcohol_status": alcohol_status})
    
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    GPIO.cleanup()
    cam.release()
    cv2.destroyAllWindows()
    


