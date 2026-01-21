"""
Conversation manager - handles multiple conversations with separate documents.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass, asdict
from utils import get_writable_path


@dataclass
class Conversation:
    """Represents a single conversation."""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    document_ids: List[str]  # List of document filenames
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Conversation':
        return cls(**data)


class ConversationManager:
    """Manages multiple conversations with their documents."""
    
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        self.current_id: Optional[str] = None
        
        # Paths
        self.data_dir = get_writable_path("conversations")
        self.index_file = os.path.join(self.data_dir, "index.json")
        self.docs_dir = get_writable_path("documents")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)
        
        self._load_index()
    
    def _load_index(self):
        """Load conversations index."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.conversations = {
                    cid: Conversation.from_dict(conv) 
                    for cid, conv in data.get("conversations", {}).items()
                }
                self.current_id = data.get("current_id")
                
                # Validate current_id
                if self.current_id and self.current_id not in self.conversations:
                    self.current_id = None
                    
            except Exception as e:
                print(f"[ConvManager] Error loading index: {e}")
                self.conversations = {}
                self.current_id = None
    
    def _save_index(self):
        """Save conversations index."""
        try:
            data = {
                "conversations": {cid: conv.to_dict() for cid, conv in self.conversations.items()},
                "current_id": self.current_id
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ConvManager] Error saving index: {e}")
    
    def _generate_id(self) -> str:
        """Generate unique conversation ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def _generate_title(self, first_message: str = "") -> str:
        """Generate conversation title from first message."""
        if first_message:
            # Take first 40 chars of first message
            title = first_message[:40].strip()
            if len(first_message) > 40:
                title += "..."
            return title
        return f"New Chat {datetime.now().strftime('%d/%m %H:%M')}"
    
    def create_conversation(self, title: str = "") -> Conversation:
        """Create a new conversation."""
        conv_id = self._generate_id()
        now = datetime.now().isoformat()
        
        conv = Conversation(
            id=conv_id,
            title=title or self._generate_title(),
            created_at=now,
            updated_at=now,
            messages=[],
            document_ids=[]
        )
        
        self.conversations[conv_id] = conv
        self.current_id = conv_id
        self._save_index()
        
        print(f"[ConvManager] Created conversation: {conv_id}")
        return conv
    
    def get_current(self) -> Optional[Conversation]:
        """Get current conversation."""
        if self.current_id and self.current_id in self.conversations:
            return self.conversations[self.current_id]
        return None
    
    def set_current(self, conv_id: str) -> Optional[Conversation]:
        """Set current conversation."""
        if conv_id in self.conversations:
            self.current_id = conv_id
            self._save_index()
            return self.conversations[conv_id]
        return None
    
    def get_all(self) -> List[Conversation]:
        """Get all conversations sorted by updated_at (newest first)."""
        return sorted(
            self.conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True
        )
    
    def add_message(self, role: str, content: str):
        """Add message to current conversation."""
        conv = self.get_current()
        if not conv:
            conv = self.create_conversation()
        
        conv.messages.append({"role": role, "content": content})
        conv.updated_at = datetime.now().isoformat()
        
        # Update title from first user message
        if role == "user" and len(conv.messages) == 1:
            conv.title = self._generate_title(content)
        
        self._save_index()
    
    def add_document(self, filename: str):
        """Add document to current conversation."""
        conv = self.get_current()
        if not conv:
            conv = self.create_conversation()
        
        if filename not in conv.document_ids:
            conv.document_ids.append(filename)
            conv.updated_at = datetime.now().isoformat()
            self._save_index()
    
    def get_conversation_docs_folder(self, conv_id: str = None) -> str:
        """Get documents folder for a conversation."""
        cid = conv_id or self.current_id
        if not cid:
            return self.docs_dir
        
        conv_docs = os.path.join(self.docs_dir, cid)
        os.makedirs(conv_docs, exist_ok=True)
        return conv_docs
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and its documents."""
        if conv_id not in self.conversations:
            return False
        
        # Delete documents folder
        conv_docs = os.path.join(self.docs_dir, conv_id)
        if os.path.exists(conv_docs):
            try:
                shutil.rmtree(conv_docs)
            except Exception as e:
                print(f"[ConvManager] Error deleting docs: {e}")
        
        # Remove from index
        del self.conversations[conv_id]
        
        # Update current if needed
        if self.current_id == conv_id:
            remaining = self.get_all()
            self.current_id = remaining[0].id if remaining else None
        
        self._save_index()
        print(f"[ConvManager] Deleted conversation: {conv_id}")
        return True
    
    def clear_history(self):
        """Clear messages in current conversation."""
        conv = self.get_current()
        if conv:
            conv.messages = []
            conv.updated_at = datetime.now().isoformat()
            self._save_index()
    
    def rename_conversation(self, conv_id: str, new_title: str) -> bool:
        """Rename a conversation."""
        if conv_id in self.conversations:
            self.conversations[conv_id].title = new_title
            self.conversations[conv_id].updated_at = datetime.now().isoformat()
            self._save_index()
            return True
        return False
    
    def cleanup_orphan_documents(self):
        """Remove documents not linked to any conversation."""
        # Get all document IDs from all conversations
        all_doc_ids = set()
        for conv in self.conversations.values():
            all_doc_ids.update(conv.document_ids)
        
        # Check root docs folder for orphans
        if os.path.exists(self.docs_dir):
            for item in os.listdir(self.docs_dir):
                item_path = os.path.join(self.docs_dir, item)
                
                # Skip conversation folders
                if os.path.isdir(item_path) and item in self.conversations:
                    continue
                
                # Remove orphan files in root
                if os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                        print(f"[ConvManager] Removed orphan: {item}")
                    except Exception as e:
                        print(f"[ConvManager] Error removing {item}: {e}")
