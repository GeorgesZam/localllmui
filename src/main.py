import tkinter as tk
from tkinter import scrolledtext
import threading
import os
import sys
import traceback

 
def resource_path(relative_path):
    """Chemin compatible PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")
    return os.path.join(base, relative_path)


class LocalChat:
    def __init__(self, root):
        self.root = root
        self.root.title("Local Chat - Debug")
        self.root.geometry("800x600")
        self.root.configure(bg="#1a1a2e")
        
        self.llm = None
        self.is_generating = False
        
        self.create_ui()
        self.load_model()
    
    def create_ui(self):
        bg = "#1a1a2e"
        fg = "#eaeaea"
        accent = "#4a9eff"
        
        header = tk.Frame(self.root, bg=bg)
        header.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(header, text="🤖 Local Chat", font=("Arial", 20, "bold"), bg=bg, fg=fg).pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header, text="⏳ Chargement...", font=("Arial", 11), bg=bg, fg="#888")
        self.status_label.pack(side=tk.RIGHT)
        
        chat_frame = tk.Frame(self.root, bg=bg)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.chat = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, font=("Consolas", 11),
            bg="#16213e", fg=fg, insertbackground=fg, relief=tk.FLAT, padx=10, pady=10
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.config(state=tk.DISABLED)
        
        self.chat.tag_configure("user", foreground=accent, font=("Consolas", 11, "bold"))
        self.chat.tag_configure("bot", foreground="#50fa7b", font=("Consolas", 11, "bold"))
        self.chat.tag_configure("system", foreground="#888", font=("Consolas", 10, "italic"))
        self.chat.tag_configure("error", foreground="#ff5555", font=("Consolas", 10))
        self.chat.tag_configure("debug", foreground="#ffb86c", font=("Consolas", 9))
        
        input_frame = tk.Frame(self.root, bg=bg)
        input_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.input_box = tk.Text(
            input_frame, height=2, font=("Consolas", 11),
            bg="#16213e", fg=fg, insertbackground=fg, relief=tk.FLAT, padx=10, pady=8
        )
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_box.bind("<Return>", self.on_enter)
        
        self.send_btn = tk.Button(
            input_frame, text="Envoyer", font=("Arial", 11, "bold"),
            bg=accent, fg="white", relief=tk.FLAT, padx=20, pady=8, command=self.send
        )
        self.send_btn.pack(side=tk.RIGHT)
        
        self.add_message("Système", "Démarrage...", "system")
    
    def load_model(self):
        def _load():
            try:
                # Debug: afficher les chemins
                self.root.after(0, lambda: self.add_message("Debug", f"sys._MEIPASS existe: {hasattr(sys, '_MEIPASS')}", "debug"))
                
                if hasattr(sys, '_MEIPASS'):
                    self.root.after(0, lambda: self.add_message("Debug", f"_MEIPASS: {sys._MEIPASS}", "debug"))
                
                self.root.after(0, lambda: self.add_message("Debug", f"CWD: {os.getcwd()}", "debug"))
                self.root.after(0, lambda: self.add_message("Debug", f"EXE: {sys.executable}", "debug"))
                
                # Lister le contenu de _MEIPASS ou CWD
                base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.abspath(".")
                self.root.after(0, lambda: self.add_message("Debug", f"Base path: {base_path}", "debug"))
                
                try:
                    files = os.listdir(base_path)
                    self.root.after(0, lambda: self.add_message("Debug", f"Contenu base: {files[:10]}", "debug"))
                except Exception as e:
                    self.root.after(0, lambda: self.add_message("Debug", f"Erreur listdir base: {e}", "error"))
                
                # Chercher le dossier models
                models_path = os.path.join(base_path, "models")
                self.root.after(0, lambda: self.add_message("Debug", f"Models path: {models_path}", "debug"))
                self.root.after(0, lambda: self.add_message("Debug", f"Models existe: {os.path.exists(models_path)}", "debug"))
                
                if os.path.exists(models_path):
                    try:
                        model_files = os.listdir(models_path)
                        self.root.after(0, lambda: self.add_message("Debug", f"Contenu models: {model_files}", "debug"))
                    except Exception as e:
                        self.root.after(0, lambda: self.add_message("Debug", f"Erreur listdir models: {e}", "error"))
                
                # Import llama_cpp
                self.root.after(0, lambda: self.add_message("Debug", "Import llama_cpp...", "debug"))
                from llama_cpp import Llama
                self.root.after(0, lambda: self.add_message("Debug", "Import OK", "debug"))
                
                # Chemin du modèle
                model_file = resource_path("models/model.gguf")
                self.root.after(0, lambda: self.add_message("Debug", f"Model file: {model_file}", "debug"))
                self.root.after(0, lambda: self.add_message("Debug", f"Model existe: {os.path.exists(model_file)}", "debug"))
                
                if not os.path.exists(model_file):
                    self.root.after(0, lambda: self.status_label.config(text="❌ Modèle introuvable"))
                    self.root.after(0, lambda: self.add_message("Erreur", f"Fichier non trouvé: {model_file}", "error"))
                    return
                
                # Taille du fichier
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                self.root.after(0, lambda: self.add_message("Debug", f"Taille modèle: {size_mb:.1f} MB", "debug"))
                
                # Charger le modèle
                self.root.after(0, lambda: self.add_message("Debug", "Chargement du modèle...", "debug"))
                
                self.llm = Llama(
                    model_path=model_file,
                    n_ctx=1024,
                    n_threads=4,
                    verbose=False
                )
                
                self.root.after(0, lambda: self.status_label.config(text="✅ Prêt"))
                self.root.after(0, lambda: self.add_message("Système", "Modèle chargé ! Posez vos questions.", "system"))
                
            except Exception as e:
                error_msg = traceback.format_exc()
                self.root.after(0, lambda: self.status_label.config(text="❌ Erreur"))
                self.root.after(0, lambda: self.add_message("Erreur", error_msg, "error"))
        
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
        if not (event.state & 0x1):
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
        self.root.after(0, lambda: self.add_message("Assistant", "", "bot"))
        
        try:
            prompt = f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
            
            for chunk in self.llm(prompt, max_tokens=256, stop=["<|im_end|>"], stream=True):
                token = chunk["choices"][0]["text"]
                self.root.after(0, lambda t=token: self.stream_text(t))
                
        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: self.add_message("Erreur", error_msg, "error"))
        finally:
            self.root.after(0, self.done)
    
    def done(self):
        self.is_generating = False
        self.send_btn.config(state=tk.NORMAL)
        self.input_box.focus_set()


if __name__ == "__main__":
    root = tk.Tk()
    app = LocalChat(root)
    root.mainloop()
