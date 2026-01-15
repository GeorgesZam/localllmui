import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import os
import sys


def resource_path(relative_path):
    """Chemin compatible PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class GemmaChat:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemma 3 - Local Chat")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1a2e")
        
        self.llm = None
        self.is_generating = False
        
        self.create_ui()
        self.load_model()
    
    def create_ui(self):
        # Couleurs
        bg = "#1a1a2e"
        fg = "#eaeaea"
        accent = "#4a9eff"
        
        # Header
        header = tk.Frame(self.root, bg=bg)
        header.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(
            header, 
            text="💎 Gemma 3 Local", 
            font=("Arial", 20, "bold"),
            bg=bg, 
            fg=fg
        ).pack(side=tk.LEFT)
        
        self.status_label = tk.Label(
            header,
            text="⏳ Chargement...",
            font=("Arial", 11),
            bg=bg,
            fg="#888"
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Zone de chat
        chat_frame = tk.Frame(self.root, bg=bg)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.chat = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#16213e",
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.config(state=tk.DISABLED)
        
        # Tags
        self.chat.tag_configure("user", foreground=accent, font=("Consolas", 11, "bold"))
        self.chat.tag_configure("bot", foreground="#50fa7b", font=("Consolas", 11, "bold"))
        self.chat.tag_configure("system", foreground="#888", font=("Consolas", 10, "italic"))
        
        # Zone input
        input_frame = tk.Frame(self.root, bg=bg)
        input_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.input_box = tk.Text(
            input_frame,
            height=2,
            font=("Consolas", 11),
            bg="#16213e",
            fg=fg,
            insertbackground=fg,
            relief=tk.FLAT,
            padx=10,
            pady=8
        )
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_box.bind("<Return>", self.on_enter)
        
        self.send_btn = tk.Button(
            input_frame,
            text="Envoyer",
            font=("Arial", 11, "bold"),
            bg=accent,
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=self.send
        )
        self.send_btn.pack(side=tk.RIGHT)
        
        self.add_message("Système", "Chargement de Gemma 3...", "system")
    
    def load_model(self):
        def _load():
            try:
                from llama_cpp import Llama
                
                model_file = resource_path("models/gemma3.gguf")
                
                if not os.path.exists(model_file):
                    self.root.after(0, lambda: self.status_label.config(text="❌ Modèle introuvable"))
                    self.root.after(0, lambda: self.add_message("Système", f"Erreur: {model_file} non trouvé", "system"))
                    return
                
                self.llm = Llama(
                    model_path=model_file,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                
                self.root.after(0, lambda: self.status_label.config(text="✅ Prêt"))
                self.root.after(0, lambda: self.add_message("Système", "Gemma 3 chargé ! Posez vos questions.", "system"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text="❌ Erreur"))
                self.root.after(0, lambda: self.add_message("Système", f"Erreur: {str(e)}", "system"))
        
        threading.Thread(target=_load, daemon=True).start()
    
    def add_message(self, sender, text, tag=""):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"\n{sender}:\n", tag)
        self.chat.insert(tk.END, f"{text}\n")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)
    
    def stream_text(self, text):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text)
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)
    
    def on_enter(self, event):
        if not (event.state & 0x1):  # Shift not pressed
            self.send()
            return "break"
    
    def send(self):
        if self.is_generating or not self.llm:
            return
        
        msg = self.input_box.get("1.0", tk.END).strip()
        if not msg:
            return
        
        self.input_box.delete("1.0", tk.END)
        self.add_message("Vous", msg, "user")
        
        self.is_generating = True
        self.send_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self.generate, args=(msg,), daemon=True).start()
    
    def generate(self, msg):
        self.root.after(0, lambda: self.add_message("Gemma", "", "bot"))
        
        try:
            prompt = f"<start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n"
            
            for chunk in self.llm(
                prompt,
                max_tokens=512,
                stop=["<end_of_turn>"],
                stream=True
            ):
                token = chunk["choices"][0]["text"]
                self.root.after(0, lambda t=token: self.stream_text(t))
                
        except Exception as e:
            self.root.after(0, lambda: self.add_message("Système", f"Erreur: {str(e)}", "system"))
        finally:
            self.root.after(0, self.done)
    
    def done(self):
        self.is_generating = False
        self.send_btn.config(state=tk.NORMAL)
        self.input_box.focus_set()


if __name__ == "__main__":
    root = tk.Tk()
    app = GemmaChat(root)
    root.mainloop()