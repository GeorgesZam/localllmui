"""
Skills Manager for Local RAG Application
Manages available skills and their configuration.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image


@dataclass
class SkillInfo:
    """Information about a skill."""

    id: str
    name: str
    description: str
    category: str
    icon: str
    enabled: bool = False
    config: dict = None
    image_path: Optional[str] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


class SkillsManager:
    """Manages loading and saving skill configurations."""

    def __init__(self):
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.parent
        self.skills_dir = script_dir / "skills"
        self.config_file = script_dir / "skills_config.json"
        self.skills: Dict[str, SkillInfo] = {}
        self._load_skills()

    def _load_skills(self):
        """Load all skills from the skills directory."""
        if not self.skills_dir.exists():
            return

        for skill_file in self.skills_dir.glob("skill_*.md"):
            skill_id = skill_file.stem.replace("skill_", "")
            self.skills[skill_id] = self._parse_skill_file(skill_file)

        self._load_config()

    def _parse_skill_file(self, skill_file: Path) -> SkillInfo:
        """Parse a skill markdown file to extract metadata."""
        content = skill_file.read_text()
        lines = content.split("\n")

        skill_id = skill_file.stem.replace("skill_", "")
        name = skill_id.replace("_", " ").title()
        description = ""
        category = "General"
        icon = "🔧"

        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                name = line.replace("# ", "").strip()
                break
            elif line.startswith("## Description:"):
                description = line.replace("## Description:", "").strip()
            elif line.startswith("## Use Cases:"):
                break

        category_map = {
            "docx": "Documents",
            "pdf": "Documents",
            "ocr": "Documents",
            "rag": "AI",
            "summary": "AI",
            "manim": "Visualization",
            "code": "Development",
        }

        icon_map = {
            "docx": "📘",
            "pdf": "📕",
            "ocr": "🔍",
            "rag": "🧠",
            "summary": "📝",
            "manim": "🎬",
            "code": "💻",
        }

        for key, cat in category_map.items():
            if key in skill_id:
                category = cat
                icon = icon_map.get(key, "🔧")
                break

        return SkillInfo(
            id=skill_id,
            name=name,
            description=(
                description[:100] + "..." if len(description) > 100 else description
            ),
            category=category,
            icon=icon,
            enabled=False,
        )

    def _load_config(self):
        """Load skill configurations from JSON."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    for skill_id, enabled in config.get("enabled_skills", {}).items():
                        if skill_id in self.skills:
                            self.skills[skill_id].enabled = enabled
            except Exception as e:
                print(f"[Skills] Error loading config: {e}")

    def save_config(self):
        """Save skill configurations to JSON."""
        config = {
            "enabled_skills": {
                skill_id: skill.enabled
                for skill_id, skill in self.skills.items()
                if skill.enabled
            }
        }

        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[Skills] Error saving config: {e}")

    def get_all_skills(self) -> List[SkillInfo]:
        """Get all skills sorted by category and name."""
        return sorted(self.skills.values(), key=lambda s: (s.category, s.name))

    def get_skill_content(self, skill_id: str) -> Optional[str]:
        """Get the full content of a skill file."""
        skill_file = self.skills_dir / f"skill_{skill_id}.md"
        if skill_file.exists():
            return skill_file.read_text()
        return None

    def get_skill_instructions(self, skill_id: str) -> Optional[str]:
        """Extract the instructions section from a skill file."""
        content = self.get_skill_content(skill_id)
        if not content:
            return None

        lines = content.split("\n")
        instructions = []
        in_instructions = False

        for line in lines:
            if line.startswith("## Instructions:"):
                in_instructions = True
                continue
            if in_instructions:
                instructions.append(line)

        return "\n".join(instructions).strip() if instructions else None

    def create_skill(
        self,
        name: str,
        description: str,
        category: str,
        icon: str,
        content: str,
        image_path: Optional[str] = None,
    ) -> Optional[str]:
        """Create a new skill."""
        if not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True, exist_ok=True)

        skill_id = name.lower().replace(" ", "_").replace("-", "_")
        skill_id = "".join(c for c in skill_id if c.isalnum() or c == "_")

        if skill_id in self.skills:
            return None

        skill_file = self.skills_dir / f"skill_{skill_id}.md"
        skill_content = f"""# {name}

## Description:
{description}

## Use Cases:
- Add specific use cases here

## Instructions:
{content}
"""

        try:
            skill_file.write_text(skill_content)

            saved_image_path = None
            if image_path:
                image_ext = Path(image_path).suffix
                dest_image = self.skills_dir / f"skill_{skill_id}_icon{image_ext}"
                shutil.copy2(image_path, dest_image)
                saved_image_path = str(dest_image)

            self.skills[skill_id] = SkillInfo(
                id=skill_id,
                name=name,
                description=description,
                category=category,
                icon=icon,
                enabled=False,
                image_path=saved_image_path,
            )

            self.save_config()
            return skill_id

        except Exception as e:
            print(f"[Skills] Error creating skill: {e}")
            return None

    def update_skill(
        self,
        skill_id: str,
        name: str,
        description: str,
        category: str,
        icon: str,
        content: str,
        image_path: Optional[str] = None,
    ) -> bool:
        """Update an existing skill."""
        if skill_id not in self.skills:
            return False

        try:
            skill_file = self.skills_dir / f"skill_{skill_id}.md"
            skill_content = f"""# {name}

## Description:
{description}

## Use Cases:
- Add specific use cases here

## Instructions:
{content}
"""
            skill_file.write_text(skill_content)

            # Handle image update
            skill = self.skills[skill_id]
            old_image_path = skill.image_path

            saved_image_path = old_image_path
            if image_path and image_path != old_image_path:
                # Delete old image if exists
                if old_image_path:
                    old_path = Path(old_image_path)
                    if old_path.exists():
                        old_path.unlink()

                # Save new image
                image_ext = Path(image_path).suffix
                dest_image = self.skills_dir / f"skill_{skill_id}_icon{image_ext}"
                shutil.copy2(image_path, dest_image)
                saved_image_path = str(dest_image)

            # Update skill info
            skill.name = name
            skill.description = description
            skill.category = category
            skill.icon = icon
            skill.image_path = saved_image_path

            self.save_config()
            return True

        except Exception as e:
            print(f"[Skills] Error updating skill: {e}")
            return False

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill."""
        if skill_id not in self.skills:
            return False

        try:
            skill_file = self.skills_dir / f"skill_{skill_id}.md"
            if skill_file.exists():
                skill_file.unlink()

            skill = self.skills[skill_id]
            if skill.image_path:
                image_path = Path(skill.image_path)
                if image_path.exists():
                    image_path.unlink()

            del self.skills[skill_id]
            self.save_config()
            return True

        except Exception as e:
            print(f"[Skills] Error deleting skill: {e}")
            return False


class SkillsPanel(ctk.CTkFrame):
    """UI panel for managing skills."""

    def __init__(self, parent, skills_manager: SkillsManager, on_skill_toggle=None):
        super().__init__(parent, fg_color="#1a1a2e", corner_radius=0)

        self.skills_manager = skills_manager
        self.on_skill_toggle = on_skill_toggle

        self._create_widgets()
        self._load_skills()

    def _create_widgets(self):
        """Create the skills panel UI."""
        header = ctk.CTkFrame(self, fg_color="transparent", height=60)
        header.pack(fill="x", padx=15, pady=(15, 10))
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🎯 Skills",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#4a9eff",
        ).pack(side="left", pady=10)

        ctk.CTkButton(
            header,
            text="+ Create Skill",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#50fa7b",
            hover_color="#40c969",
            text_color="#000000",
            width=100,
            height=32,
            corner_radius=8,
            command=self._open_create_skill_dialog,
        ).pack(side="right", pady=10)

        self.skills_container = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color="#4a9eff",
            scrollbar_button_hover_color="#3b7ac7",
        )
        self.skills_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _load_skills(self):
        """Load and display all skills."""
        for widget in self.skills_container.winfo_children():
            widget.destroy()

        categories = {}
        for skill in self.skills_manager.get_all_skills():
            if skill.category not in categories:
                categories[skill.category] = []
            categories[skill.category].append(skill)

        for category, skills in categories.items():
            category_frame = ctk.CTkFrame(
                self.skills_container, fg_color="#252535", corner_radius=10
            )
            category_frame.pack(fill="x", pady=5)

            cat_header = ctk.CTkFrame(category_frame, fg_color="transparent")
            cat_header.pack(fill="x", padx=10, pady=(8, 5))

            ctk.CTkLabel(
                cat_header,
                text=f"📁 {category}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#888888",
            ).pack(side="left")

            ctk.CTkLabel(
                cat_header,
                text=f"{len(skills)} skills",
                font=ctk.CTkFont(size=10),
                text_color="#666666",
            ).pack(side="right")

            for skill in skills:
                self._create_skill_item(category_frame, skill).pack(
                    fill="x", padx=10, pady=(0, 5)
                )

    def _create_skill_item(self, parent, skill: SkillInfo) -> ctk.CTkFrame:
        """Create a skill item widget."""
        item = ctk.CTkFrame(
            parent,
            fg_color="#1e1e2e" if not skill.enabled else "#2a3a2a",
            corner_radius=8,
            border_width=1,
            border_color="#3a3a4a",
        )
        item.pack(fill="x", pady=2)

        content = ctk.CTkFrame(item, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=10, pady=8)

        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x")

        left_side = ctk.CTkFrame(top_row, fg_color="transparent")
        left_side.pack(side="left")

        if skill.image_path and Path(skill.image_path).exists():
            try:
                pil_image = Image.open(skill.image_path)
                pil_image = pil_image.resize((20, 20), Image.Resampling.LANCZOS)
                icon_label = ctk.CTkLabel(
                    left_side, image=ctk.CTkImage(pil_image, size=(20, 20)), text=""
                )
            except Exception:
                icon_label = ctk.CTkLabel(
                    left_side, text=skill.icon, font=ctk.CTkFont(size=14)
                )
        else:
            icon_label = ctk.CTkLabel(
                left_side, text=skill.icon, font=ctk.CTkFont(size=14)
            )

        icon_label.pack(side="left", padx=(0, 5))

        name_label = ctk.CTkLabel(
            left_side,
            text=skill.name,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff" if skill.enabled else "#aaaaaa",
        )
        name_label.pack(side="left")

        if not skill.enabled:
            ctk.CTkButton(
                top_row,
                text="✕",
                width=24,
                height=24,
                corner_radius=4,
                fg_color="transparent",
                hover_color="#ff5555",
                font=ctk.CTkFont(size=10),
                command=lambda s=skill.id: self._delete_skill(s),
            ).pack(side="right", padx=(0, 5))

        toggle = ctk.CTkSwitch(
            top_row,
            text="",
            width=40,
            height=20,
            progress_color="#4a9eff" if not skill.enabled else "#50fa7b",
            button_color="#4a9eff" if not skill.enabled else "#50fa7b",
            button_hover_color="#3b7ac7" if not skill.enabled else "#40c969",
            fg_color="transparent",
        )
        toggle.pack(side="right")
        toggle.select() if skill.enabled else toggle.deselect()

        def toggle_skill():
            is_enabled = toggle.get()
            skill.enabled = is_enabled

            item.configure(
                fg_color="#2a3a2a" if is_enabled else "#1e1e2e",
                border_color="#4a9eff" if is_enabled else "#3a3a4a",
            )
            name_label.configure(text_color="#ffffff" if is_enabled else "#aaaaaa")
            toggle.configure(
                progress_color="#50fa7b" if is_enabled else "#4a9eff",
                button_color="#50fa7b" if is_enabled else "#4a9eff",
                button_hover_color="#40c969" if is_enabled else "#3b7ac7",
            )

            self.skills_manager.save_config()

            if self.on_skill_toggle:
                self.on_skill_toggle(skill.id, is_enabled)

        toggle.configure(command=toggle_skill)

        if skill.description and skill.description != "...":
            ctk.CTkLabel(
                content,
                text=skill.description,
                font=ctk.CTkFont(size=10),
                text_color="#888888",
                anchor="w",
            ).pack(fill="x", pady=(3, 0))

        return item

    def _delete_skill(self, skill_id: str):
        """Delete a skill after confirmation."""
        skill = self.skills_manager.skills.get(skill_id)
        if not skill:
            return

        if messagebox.askyesno(
            "Delete Skill",
            f"Are you sure you want to delete '{skill.name}'?\n\nThis action cannot be undone.",
        ):
            if self.skills_manager.delete_skill(skill_id):
                self._load_skills()
                if self.on_skill_toggle:
                    self.on_skill_toggle(skill_id, False)
            else:
                messagebox.showerror("Error", "Failed to delete skill.")

    def _open_create_skill_dialog(self):
        """Open the create skill dialog."""
        CreateSkillDialog(
            self.winfo_toplevel(),
            self.skills_manager,
            on_skill_created=self._load_skills,
        )


class CreateSkillDialog(ctk.CTkToplevel):
    """Dialog for creating a new skill."""

    def __init__(self, parent, skills_manager: SkillsManager, on_skill_created=None):
        super().__init__(parent)

        self.skills_manager = skills_manager
        self.on_skill_created = on_skill_created
        self.selected_image_path = None

        self.title("Create New Skill")
        self.geometry("600x800")
        self.configure(fg_color="#1a1a2e")
        self.protocol("WM_DELETE_WINDOW", self._close_window)

        # Make window resizable
        self.minsize(550, 700)
        self.resizable(True, True)

        self._create_widgets()
        self._center_window()

        try:
            self.transient(parent)
            self.grab_set()
        except Exception:
            pass

    def _center_window(self):
        """Center the dialog on the parent window."""
        self.update_idletasks()
        parent = self.master
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _close_window(self):
        """Properly close the dialog and release grab."""
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _create_widgets(self):
        """Create the dialog widgets."""
        # Configure grid layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # Make scrollable row expandable
        self.rowconfigure(2, weight=0)  # Button row fixed

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent", height=60)
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(30, 10))
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="✨ Create New Skill",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#4a9eff",
        ).pack(pady=10)

        # Scrollable container for form
        scroll_container = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color="#4a9eff",
            scrollbar_button_hover_color="#3b7ac7",
            height=450  # Fixed height for scroll area
        )
        scroll_container.grid(row=1, column=0, sticky="nsew", padx=30, pady=(0, 5))

        form = ctk.CTkFrame(scroll_container, fg_color="#252535", corner_radius=15)
        form.pack(fill="x", pady=(0, 20))

        self._create_form_field(form, "Skill Name *", "e.g., Image Generator", 20)
        self.name_entry = self.last_entry

        self._create_form_field(
            form, "Description *", "Brief description of what this skill does", 15
        )
        self.desc_entry = self.last_entry

        # Category label
        ctk.CTkLabel(
            form,
            text="Category *",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(fill="x", padx=20, pady=(15, 5))

        # Category combobox
        self.category_combo = ctk.CTkComboBox(
            form,
            values=[
                "General",
                "AI",
                "Documents",
                "Development",
                "Visualization",
                "Tools",
            ],
            font=ctk.CTkFont(size=12),
            fg_color="#1e1e2e",
            button_color="#4a9eff",
            border_color="#3a3a4a",
            dropdown_fg_color="#252535",
            text_color="#ffffff",
            height=40,
        )
        self.category_combo.pack(fill="x", padx=20)
        self.category_combo.set("General")

        self._create_form_field(form, "Icon Emoji", "🔧", 15)
        self.icon_entry = self.last_entry

        icon_frame = ctk.CTkFrame(form, fg_color="#1e1e2e")
        icon_frame.pack(fill="x", padx=20, pady=(5, 0))

        for icon in ["🔧", "🎯", "📝", "🧠", "💻", "🎨", "📊", "🔍", "⚡", "🚀"]:
            ctk.CTkButton(
                icon_frame,
                text=icon,
                width=35,
                height=35,
                font=ctk.CTkFont(size=14),
                fg_color="#3a3a4a",
                hover_color="#4a9eff",
                text_color="#ffffff",
                corner_radius=8,
                command=lambda i=icon: self._select_icon(i),
            ).pack(side="left", padx=2, pady=5)

        ctk.CTkLabel(
            form,
            text="Skill Icon (Image)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(fill="x", padx=20, pady=(15, 5))

        image_btn_frame = ctk.CTkFrame(form, fg_color="transparent")
        image_btn_frame.pack(fill="x", padx=20)

        ctk.CTkButton(
            image_btn_frame,
            text="📁 Upload Image",
            font=ctk.CTkFont(size=11),
            fg_color="#4a9eff",
            hover_color="#3b7ac7",
            width=120,
            height=35,
            corner_radius=8,
            command=self._upload_image,
        ).pack(side="left", padx=(0, 10))

        self.image_status = ctk.CTkLabel(
            image_btn_frame,
            text="No image selected",
            font=ctk.CTkFont(size=10),
            text_color="#666666",
        )
        self.image_status.pack(side="left", pady=5)

        ctk.CTkLabel(
            form,
            text="Instructions / Content *",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(fill="x", padx=20, pady=(15, 5))

        self.content_text = ctk.CTkTextbox(
            form,
            font=ctk.CTkFont(size=11),
            fg_color="#1e1e2e",
            border_color="#3a3a4a",
            text_color="#ffffff",
            height=150,
        )
        self.content_text.pack(fill="x", padx=20, pady=(0, 20))

        # Buttons at bottom (always visible)
        button_frame = ctk.CTkFrame(self, fg_color="#252535", corner_radius=15)
        button_frame.grid(row=2, column=0, sticky="ew", padx=30, pady=(5, 25))

        # Center the buttons
        button_inner = ctk.CTkFrame(button_frame, fg_color="transparent")
        button_inner.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkButton(
            button_inner,
            text="✕ Cancel",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3a3a4a",
            hover_color="#4a4a5a",
            text_color="#ffffff",
            width=140,
            height=45,
            corner_radius=10,
            command=self._close_window,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_inner,
            text="✓ Create Skill",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#50fa7b",
            hover_color="#40c969",
            text_color="#000000",
            width=150,
            height=45,
            corner_radius=10,
            command=self._create_skill,
        ).pack(side="left", padx=10)

    def _create_form_field(self, parent, label_text, placeholder, top_padding):
        """Helper to create form labels and entries."""
        ctk.CTkLabel(
            parent,
            text=label_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff",
            anchor="w",
        ).pack(fill="x", padx=20, pady=(top_padding, 5))

        self.last_entry = ctk.CTkEntry(
            parent,
            placeholder_text=placeholder,
            font=ctk.CTkFont(size=12),
            fg_color="#1e1e2e",
            border_color="#3a3a4a",
            text_color="#ffffff",
            placeholder_text_color="#666666",
            height=40,
        )
        self.last_entry.pack(fill="x", padx=20)

    def _select_icon(self, icon: str):
        """Select an icon from the quick picker."""
        self.icon_entry.delete(0, "end")
        self.icon_entry.insert(0, icon)

    def _upload_image(self):
        """Open file dialog to upload an image."""
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Select Skill Icon",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.selected_image_path = file_path
            filename = Path(file_path).name
            self.image_status.configure(
                text=f"✓ {filename[:30]}..." if len(filename) > 30 else f"✓ {filename}",
                text_color="#50fa7b",
            )

    def _create_skill(self):
        """Create the new skill."""
        name = self.name_entry.get().strip()
        description = self.desc_entry.get().strip()
        category = self.category_combo.get()
        icon = self.icon_entry.get().strip() or "🔧"
        content = self.content_text.get("1.0", "end").strip()

        if not name:
            messagebox.showerror("Validation Error", "Please enter a skill name.")
            return

        if not description:
            messagebox.showerror("Validation Error", "Please enter a description.")
            return

        if not content:
            messagebox.showerror("Validation Error", "Please enter skill instructions.")
            return

        skill_id = self.skills_manager.create_skill(
            name=name,
            description=description,
            category=category,
            icon=icon,
            content=content,
            image_path=self.selected_image_path,
        )

        if skill_id:
            messagebox.showinfo("Success", f"Skill '{name}' created successfully!")
            if self.on_skill_created:
                self.on_skill_created()
            self._close_window()
        else:
            messagebox.showerror(
                "Error", "Failed to create skill. It may already exist."
            )


class SkillExecutor:
    """Executes enabled skills during conversation."""

    def __init__(self, skills_manager: SkillsManager):
        self.skills_manager = skills_manager

    def apply_skills_to_prompt(
        self, message: str, base_prompt: str
    ) -> tuple[str, list]:
        """Apply enabled skills to enhance the prompt."""
        enhanced_prompts = []
        skill_names = []

        for skill_id, skill in self.skills_manager.skills.items():
            if skill.enabled:
                skill_content = self.skills_manager.get_skill_content(skill_id)
                if skill_content:
                    enhanced_prompts.append(f"# {skill.name}\n{skill_content}\n")
                    skill_names.append(skill.name)

        if enhanced_prompts:
            skills_section = "\n".join(enhanced_prompts)
            enhanced_prompt = f"""{base_prompt}

=== AVAILABLE SKILLS ===
{skill_names}

{skills_section}

"""
            return enhanced_prompt, skill_names

        return base_prompt, []
