import sys
import tkinter as tk
from tkinter import scrolledtext
import threading
import time

class StdoutRedirector:
    def __init__(self, text_widget: tk.scrolledtext.ScrolledText):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        if not string:
            return
            
        def _update_ui(s=string):
            self.text_widget.configure(state=tk.NORMAL)
            if '\r' in s:
                lines = s.split('\r')
                final_text = lines[-1] if lines[-1] else (lines[-2] if len(lines) > 1 else "")
                self.text_widget.delete("end-1c linestart", "end")
                self.text_widget.insert(tk.END, final_text)
            else:
                self.text_widget.insert(tk.END, s)
                
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)
            
        self.text_widget.after(0, _update_ui)

    def flush(self):
        pass

def run_test():
    root = tk.Tk()
    text = scrolledtext.ScrolledText(root)
    text.pack()
    
    def worker():
        original = sys.stdout
        sys.stdout = StdoutRedirector(text)
        print("Hello World!")
        for i in range(5):
            sys.stdout.write(f"\rProgress: {i}/10")
            sys.stdout.flush()
            time.sleep(0.5)
        print("\nDone!")
        time.sleep(1)
        sys.stdout = original
        root.after(0, root.destroy)
        
    threading.Thread(target=worker, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    run_test()
