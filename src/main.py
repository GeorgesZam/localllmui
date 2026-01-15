import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import json
import os


class OllamaLocalUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Local UI")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")
        self.root.minsize(800, 600)
        
        self.current_model = tk.StringVar(value="llama3.2")
        self.is_generating = False
        self.conversation_history = []
        
        self.setup_styles()
        self.create_widgets()
        self.check_ollama_status()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        self.bg_dark = "#1e1e1e"
        self.bg_medium = "#2d2d2d"
        self.bg_light = "#3c3c3c"
        self.fg_white = "#ffffff"
        self.fg_gray = "#b0b0b0"
        self.accent = "#4a9eff"
        self.success = "#4caf50"
        self.error = "#f44336"
        
        style.configure("TFrame", background=self.bg_dark)
        style.configure("TLabel", background=self.bg_dark, foreground=self.fg_white, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TCombobox", font=("Segoe UI", 10))
    
    def create_widgets(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        title_label = ttk.Label(header_frame, text="🤖 LLM Local UI", style="Title.TLabel")
        title_label.pack(side=tk.LEFT)
        
        self.status_frame = tk.Frame(header_frame, bg=self.bg_dark)
        self.status_frame.pack(side=tk.RIGHT)
        
        self.status_dot = tk.Canvas(self.status_frame, width=12, height=12, bg=self.bg_dark, highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 5))
        self.status_indicator = self.status_dot.create_oval(2, 2, 10, 10, fill=self.fg_gray)
        
        self.status_label = ttk.Label(self.status_frame, text="Vérification...", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        model_frame = ttk.Frame(self.root)
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(model_frame, text="Modèle:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_combo = ttk.Combobox(
            model_frame, 
            textvariable=self.current_model,
            values=["llama3.2", "llama3.1", "mistral", "codellama", "phi3", "gemma2"],
            state="readonly",
            width=20
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        self.refresh_btn = ttk.Button(model_frame, text="🔄 Rafraîchir", command=self.refresh_models)
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(model_frame, text="🗑️ Effacer", command=self.clear_conversation)
        self.clear_btn.pack(side=tk.LEFT)
        
        chat_frame = ttk.Frame(self.root)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg=self.bg_medium,
            fg=self.fg_white,
            insertbackground=self.fg_white,
            selectbackground=self.accent,
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        self.chat_display.tag_configure("user", foreground=self.accent, font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("assistant", foreground=self.success, font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("error", foreground=self.error)
        self.chat_display.tag_configure("system", foreground=self.fg_gray, font=("Consolas", 9, "italic"))
        
        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.input_text = tk.Text(
            input_frame,
            height=3,
            font=("Consolas", 11),
            bg=self.bg_light,
            fg=self.fg_white,
            insertbackground=self.fg_white,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_text.bind("<Return>", self.on_enter_pressed)
        self.input_text.bind("<Shift-Return>", lambda e: None)
        
        self.send_btn = ttk.Button(input_frame, text="Envoyer ➤", command=self.send_message, width=12)
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.append_to_chat("Système", "Bienvenue ! Assurez-vous qu'Ollama est installé et lancé.\n", "system")
    
    def check_ollama_status(self):
        def check():
            try:
                creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=creationflags
                )
                if result.returncode == 0:
                    self.root.after(0, lambda: self.update_status(True, "Ollama connecté"))
                    models = self.parse_ollama_list(result.stdout)
                    if models:
                        self.root.after(0, lambda: self.update_models(models))
                else:
                    self.root.after(0, lambda: self.update_status(False, "Ollama non disponible"))
            except FileNotFoundError:
                self.root.after(0, lambda: self.update_status(False, "Ollama non installé"))
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self.update_status(False, "Timeout"))
            except Exception as e:
                self.root.after(0, lambda: self.update_status(False, "Erreur"))
        
        threading.Thread(target=check, daemon=True).start()
    
    def parse_ollama_list(self, output):
        models = []
        lines = output.strip().split('\n')
        for line in lines[1:]:
            if line.strip():
                model_name = line.split()[0].split(':')[0]
                if model_name and model_name not in models:
                    models.append(model_name)
        return models
    
    def update_status(self, connected, message):
        color = self.success if connected else self.error
        self.status_dot.itemconfig(self.status_indicator, fill=color)
        self.status_label.config(text=message)
    
    def update_models(self, models):
        self.model_combo['values'] = models
        if models and self.current_model.get() not in models:
            self.current_model.set(models[0])
    
    def refresh_models(self):
        self.update_status(False, "Actualisation...")
        self.check_ollama_status()
    
    def clear_conversation(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.conversation_history.clear()
        self.append_to_chat("Système", "Conversation effacée.\n", "system")
    
    def append_to_chat(self, sender, message, tag=""):
        self.chat_display.config(state=tk.NORMAL)
        if sender:
            self.chat_display.insert(tk.END, f"\n{sender}:\n", tag)
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def stream_to_chat(self, text):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def on_enter_pressed(self, event):
        if not event.state & 0x1:
            self.send_message()
            return "break"
    
    def send_message(self):
        if self.is_generating:
            return
        
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return
        
        self.input_text.delete("1.0", tk.END)
        self.append_to_chat("Vous", message, "user")
        self.conversation_history.append({"role": "user", "content": message})
        
        self.is_generating = True
        self.send_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self.generate_response, args=(message,), daemon=True).start()
    
    def generate_response(self, message):
        model = self.current_model.get()
        self.root.after(0, lambda: self.append_to_chat("Assistant", "", "assistant"))
        
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen(
                ["ollama", "run", model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=creationflags
            )
            
            process.stdin.write(message + "\n")
            process.stdin.flush()
            process.stdin.close()
            
            full_response = ""
            for line in process.stdout:
                if line:
                    full_response += line
                    self.root.after(0, lambda l=line: self.stream_to_chat(l))
            
            process.wait()
            
            if full_response.strip():
                self.conversation_history.append({"role": "assistant", "content": full_response.strip()})
            
        except FileNotFoundError:
            self.root.after(0, lambda: self.append_to_chat("", "❌ Ollama n'est pas installé.\n", "error"))
        except Exception as e:
            self.root.after(0, lambda: self.append_to_chat("", f"❌ Erreur: {str(e)}\n", "error"))
        finally:
            self.root.after(0, self.generation_complete)
    
    def generation_complete(self):
        self.is_generating = False
        self.send_btn.config(state=tk.NORMAL)
        self.input_text.focus_set()


def main():
    root = tk.Tk()
    app = OllamaLocalUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()