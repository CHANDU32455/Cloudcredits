import tkinter as tk
from PIL import Image, ImageDraw

class WhiteboardCanvas(tk.Canvas):
    def __init__(self, master=None, width=280, height=280):
        super().__init__(master, width=width, height=height, bg='white', cursor='cross')
        self.width = width
        self.height = height
        self.old_x = None
        self.old_y = None

        self.image = Image.new("L", (width, height), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.bind("<B1-Motion>", self.paint)
        self.bind("<ButtonRelease-1>", self.reset)

        clear_btn = tk.Button(master, text="Clear Whiteboard", command=self.clear)
        clear_btn.pack()

    def paint(self, event):
        if self.old_x and self.old_y:
            self.create_line(self.old_x, self.old_y, event.x, event.y,
                             width=12, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.old_x, self.old_y, event.x, event.y],
                           fill=0, width=12)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def clear(self):
        self.delete("all")
        self.image = Image.new("L", (self.width, self.height), 255)
        self.draw = ImageDraw.Draw(self.image)

    def get_image(self):
        return self.image.copy()

