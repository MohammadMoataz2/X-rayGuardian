import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
model = load_model('medical.h5')
def analyze_media(media_path, result_label, media_label):
    try:
        # Load the model
      
        labels = open('labels.txt', 'r').read().splitlines()

        if media_path.lower().endswith(('.mp4', '.avi', '.mov')):
            # If it's a video file
            camera = cv2.VideoCapture(media_path)

            while True:
                ret, frame = camera.read()
                if not ret:
                    break

                image = cv2.resize(frame, (180, 180), interpolation=cv2.INTER_AREA)
                image_array = np.asarray(image, dtype=np.float32).reshape(1, 180, 180, 3) / 255.0

                probabilities = model.predict(image_array)[0]
                predicted_class = np.argmax(probabilities)
                result_text = labels[predicted_class].split(" ")[1]

                cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Media', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            camera.release()
            cv2.destroyAllWindows()
        else:
            # If it's an image file
            image = cv2.imread(media_path)
            resized_image = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)
            image_array = np.asarray(resized_image, dtype=np.float32).reshape(1, 180, 180, 3) / 255.0

            probabilities = model.predict(image_array)
            predicted_class = np.argmax(probabilities)
            result_text = labels[predicted_class].split(" ")[1]

            cv2.putText(resized_image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            image_array = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)


            image_2 = cv2.imread(media_path)
            cv2.putText(image_2, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("image",image_2)
            image = Image.fromarray(image_array)
            photo = ImageTk.PhotoImage(image=image)

            media_label.config(image=photo)
            media_label.image = photo

            result_label.config(text=f"Result: {result_text}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def browse_file(path_label, result_label, media_label):
    filename = filedialog.askopenfilename(initialdir="/", title="Select Media File",
                                          filetypes=(("Video/Image files", "*.mp4;*.avi;*.mov;*.png;*.jpg"),
                                                     ("all files", "*.*")))
    if filename:
        path_label.config(text=f"Selected Path: {filename}")
        display_media(filename, media_label)

def display_media(media_path, media_label):
    image = cv2.imread(media_path)
    resized_image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
    image_array = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_array)
    photo = ImageTk.PhotoImage(image=image)

    media_label.config(image=photo)
    media_label.image = photo

def start_analysis(path_label, result_label, media_label):
    media_path = path_label.cget("text").split(": ")[1]
    if media_path == "None":
        messagebox.showerror("Error", "Please select a media file first.")
    else:
        analyze_media(media_path, result_label, media_label)

# Create tkinter window
root = tk.Tk()
root.title("X-RayGuardian")
root.geometry("800x400")
root.configure(bg='#f0f0f0')

# Title section
title_frame = tk.Frame(root, bg='#4CAF50', padx=10, pady=10)
title_frame.pack(fill='x')

title_label = tk.Label(title_frame, text="X-RayGuardian", font=("Helvetica", 24, "bold"), bg='#4CAF50', fg='white')
title_label.pack()

desc_label = tk.Label(title_frame, text="An AI application to scan the X-ray chest if it's normal or PNEUMONIA",
                      font=("Helvetica", 14), bg='#4CAF50', fg='white')
desc_label.pack()

# Interaction section
interaction_frame = tk.Frame(root, bg='#f0f0f0', padx=20, pady=20)
interaction_frame.pack(fill='both', expand=True)


# Path label and browse button
path_label = tk.Label(interaction_frame, text="Selected Path: None", font=("Helvetica", 12), bg='#f0f0f0', anchor='w')
path_label.pack(fill='x', pady=(0, 10))

browse_button = tk.Button(interaction_frame, text="Browse Media", command=lambda: browse_file(path_label, result_label, media_label),
                          font=("Helvetica", 16), bg='#4CAF50', fg='white', padx=20, pady=10)
browse_button.pack(side="left", padx=(0, 10))

# Start analysis button
start_button = tk.Button(interaction_frame, text="Start Analysis", command=lambda: start_analysis(path_label, result_label, media_label),
                          font=("Helvetica", 16), bg='#4CAF50', fg='white', padx=20, pady=10)
start_button.pack(side="left", padx=(0, 10))

# Result label
result_label = tk.Label(interaction_frame, text="Result: ", font=("Helvetica", 12), bg='#f0f0f0', anchor='w')
result_label.pack(fill='x', pady=(10, 0))
media_label = tk.Label(interaction_frame, bg='#f0f0f0')
media_label.pack(fill='both', expand=True, pady=(0, 10))
# Start the application
root.mainloop()
