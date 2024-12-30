import sys

import tkinter as tk
import tkinter.filedialog as filedialog


class MenuGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.backgroundLoc = None
        self.videoLoc = None
        self.modelFolderLoc = None
        self.thresholdLoc = None
        self.trackingLoc = None

        self.title("Menu")
        self.geometry("550x250")
        self.configure(bg="#f2f2f2")

        # Default settings.
        self.videoFile = tk.StringVar(value=None)
        self.backgroundFile = tk.StringVar(value=None)
        self.modelFolder = tk.StringVar(value="./model/checkpoints/efficientnetb0-20231102-v0")
        self.scoreThreshold = tk.DoubleVar(value=90)
        self.trackingActivated = tk.BooleanVar(value=True)

        self.create_widgets()

    def create_widgets(self):
        self.create_file_selection("Video Data File:", self.videoFile, self.choose_input_file_video, row=0)
        self.create_file_selection("Image Background Data File (optional):", self.backgroundFile,
                                   self.choose_input_file_image, row=1)
        self.create_file_selection("Weights Folder (optional):", self.modelFolder, self.choose_model_folder, row=2)
        self.create_slider("Confidence Threshold:", self.scoreThreshold, row=3)
        self.create_checkbox("Activate Tracking:", self.trackingActivated, row=4)

        # Run button
        run_button = tk.Button(self, text="Run", command=self.ok, bg="blue", fg="white")
        run_button.grid(row=4, column=2, padx=10, pady=10)

        # Bind the closing event to the destroy method
        self.protocol("WM_DELETE_WINDOW", self.close_window)

    def create_file_selection(self, label_text, variable, command, row):
        label = tk.Label(self, text=label_text, bg="#f2f2f2")
        label.grid(row=row, column=0, padx=10, pady=10)

        entry = tk.Entry(self, textvariable=variable, width=30)
        entry.grid(row=row, column=1, padx=10, pady=10)

        button = tk.Button(self, text="Select File", command=command)
        button.grid(row=row, column=2, padx=10, pady=10)

    def create_slider(self, label_text, variable, row):
        label = tk.Label(self, text=label_text, bg="#f2f2f2")
        label.grid(row=row, column=0, padx=10, pady=10)

        slider = tk.Scale(self, variable=variable, from_=0.0, to=100.0, length=200, orient="horizontal")
        slider.grid(row=row, column=1, padx=10, pady=10)

        score_label = tk.Label(self, textvariable=variable, bg="#f2f2f2")
        score_label.grid(row=row, column=2)

        # Bind a callback to the model folder variable to enable/disable the slider
        self.modelFolder.trace_add("write", lambda *_: slider.configure(
            state=tk.NORMAL if self.modelFolder.get() else tk.DISABLED))

    def create_checkbox(self, label_text, variable, row):
        checkbox = tk.Checkbutton(self, text=label_text, variable=variable, bg="#f2f2f2")
        checkbox.grid(row=row, column=0, padx=10, pady=10, columnspan=3)

    def choose_input_file_image(self):
        self.backgroundFile.set(filedialog.askopenfilename(title="Select Image Background Data File",
                                                           filetypes=(
                                                               ("Image files", "*.png *.jpg"), ("All files", "*"))))

    def choose_input_file_video(self):
        self.videoFile.set(filedialog.askopenfilename(title="Select Video Data File",
                                                      filetypes=(("Video files", "*.avi *.mp4"), ("All files", "*"))))

    def choose_model_folder(self):
        self.modelFolder.set(filedialog.askdirectory(title="Select Model Folder"))

    def ok(self):
        self.backgroundLoc = self.backgroundFile.get()
        self.videoLoc = self.videoFile.get()
        self.modelFolderLoc = self.modelFolder.get()
        self.thresholdLoc = self.scoreThreshold.get()
        self.trackingLoc = self.trackingActivated.get()
        self.destroy()

    def close_window(self):
        # Custom logic for handling window closing
        self.destroy()
        sys.exit()
