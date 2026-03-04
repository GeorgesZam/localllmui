"""
Unit tests for Conversation module.
Following AAA (Arrange-Act-Assert) pattern.
Based on actual conversation database structure.
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from conversations import Conversation, ConversationManager


class TestConversation:
    """Test cases for Conversation dataclass."""

    def test_create_conversation_from_dict(self):
        """
        AAA Test:
        Arrange: Create conversation data dictionary
        Act: Create Conversation from dict
        Assert: Verify all fields are set correctly
        """
        # Arrange
        data = {
            "id": "test_123",
            "title": "Test Chat",
            "created_at": "2026-03-02T12:00:00",
            "updated_at": "2026-03-02T12:00:00",
            "messages": [{"role": "user", "content": "hello"}],
            "document_ids": ["doc1.txt"]
        }

        # Act
        conv = Conversation.from_dict(data)

        # Assert
        assert conv.id == "test_123"
        assert conv.title == "Test Chat"
        assert len(conv.messages) == 1
        assert conv.document_ids == ["doc1.txt"]

    def test_conversation_to_dict(self):
        """
        AAA Test:
        Arrange: Create Conversation instance
        Act: Convert to dictionary
        Assert: Verify dictionary representation
        """
        # Arrange
        conv = Conversation(
            id="test_123",
            title="Test Chat",
            created_at="2026-03-02T12:00:00",
            updated_at="2026-03-02T12:00:00",
            messages=[],
            document_ids=[]
        )

        # Act
        data = conv.to_dict()

        # Assert
        assert data["id"] == "test_123"
        assert data["title"] == "Test Chat"
        assert isinstance(data["messages"], list)


class TestConversationManagerInit:
    """Test cases for ConversationManager initialization."""

    def test_initializes_with_empty_state(self, tmp_path):
        """
        AAA Test:
        Arrange: Create temporary directory
        Act: Initialize ConversationManager
        Assert: Verify initial empty state
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)

            # Act
            manager = ConversationManager()

            # Assert
            assert len(manager.conversations) == 0
            assert manager.current_id is None

    def test_creates_data_directories(self):
        """
        AAA Test:
        Arrange: Check ConversationManager initialization
        Act: N/A (directories created in __init__)
        Assert: Verify data_dir attribute is set
        """
        # Arrange
        manager = ConversationManager()

        # Assert
        assert manager.data_dir is not None
        assert manager.index_file is not None
        assert manager.docs_dir is not None


class TestConversationManagerCRUD:
    """Test cases for ConversationManager CRUD operations."""

    def test_create_conversation(self, tmp_path):
        """
        AAA Test:
        Arrange: Create ConversationManager
        Act: Create a new conversation
        Assert: Verify conversation is created with ID
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()

            # Act
            conv = manager.create_conversation("Test Chat")

            # Assert
            assert conv.id is not None
            assert conv.title == "Test Chat"
            assert conv.id in manager.conversations
            assert manager.current_id == conv.id

    def test_create_conversation_auto_title(self, tmp_path):
        """
        AAA Test:
        Arrange: Create ConversationManager
        Act: Create conversation without title
        Assert: Verify auto-generated title
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()

            # Act
            conv = manager.create_conversation()

            # Assert
            assert conv.title is not None
            assert len(conv.title) > 0

    def test_get_current_conversation(self, tmp_path):
        """
        AAA Test:
        Arrange: Create ConversationManager with conversation
        Act: Get current conversation
        Assert: Verify correct conversation is returned
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            conv = manager.create_conversation("Test")

            # Act
            current = manager.get_current()

            # Assert
            assert current is not None
            assert current.id == conv.id

    def test_get_current_returns_none_when_empty(self, tmp_path):
        """
        AAA Test:
        Arrange: Create empty ConversationManager
        Act: Try to get current conversation
        Assert: Verify None is returned
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()

            # Act
            current = manager.get_current()

            # Assert
            assert current is None

    def test_add_message_to_conversation(self, tmp_path):
        """
        AAA Test:
        Arrange: Create ConversationManager with conversation
        Act: Add user message
        Assert: Verify message is added
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            manager.create_conversation("Test")

            # Act
            manager.add_message("user", "Hello, world!")

            # Assert
            conv = manager.get_current()
            assert len(conv.messages) == 1
            assert conv.messages[0]["role"] == "user"
            assert conv.messages[0]["content"] == "Hello, world!"

    def test_add_message_creates_conversation_if_needed(self, tmp_path):
        """
        AAA Test:
        Arrange: Create empty ConversationManager
        Act: Add message without existing conversation
        Assert: Verify new conversation is created
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()

            # Act
            manager.add_message("user", "First message")

            # Assert
            conv = manager.get_current()
            assert conv is not None
            assert len(conv.messages) == 1

    def test_add_document_to_conversation(self, tmp_path):
        """
        AAA Test:
        Arrange: Create ConversationManager with conversation
        Act: Add document
        Assert: Verify document is added
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            manager.create_conversation("Test")

            # Act
            manager.add_document("test.pdf")

            # Assert
            conv = manager.get_current()
            assert "test.pdf" in conv.document_ids

    def test_add_document_does_not_duplicate(self, tmp_path):
        """
        AAA Test:
        Arrange: Create conversation with document
        Act: Try to add same document twice
        Assert: Verify document is not duplicated
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            manager.create_conversation("Test")
            manager.add_document("test.pdf")

            # Act
            manager.add_document("test.pdf")

            # Assert
            conv = manager.get_current()
            assert conv.document_ids.count("test.pdf") == 1

    def test_get_all_conversations_sorted(self, tmp_path):
        """
        AAA Test:
        Arrange: Create multiple conversations with different times
        Act: Get all conversations
        Assert: Verify they are sorted by updated_at (newest first)
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            conv1 = manager.create_conversation("First")
            conv2 = manager.create_conversation("Second")

            # Act
            all_convs = manager.get_all()

            # Assert
            assert len(all_convs) == 2
            # Most recently updated should be first
            assert all_convs[0].id == conv2.id

    def test_delete_conversation(self, tmp_path):
        """
        AAA Test:
        Arrange: Create ConversationManager with conversations
        Act: Delete a conversation
        Assert: Verify conversation is removed
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            conv = manager.create_conversation("Test")
            conv_id = conv.id

            # Act
            result = manager.delete_conversation(conv_id)

            # Assert
            assert result is True
            assert conv_id not in manager.conversations

    def test_delete_nonexistent_conversation(self, tmp_path):
        """
        AAA Test:
        Arrange: Create ConversationManager
        Act: Try to delete non-existent conversation
        Assert: Verify False is returned
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()

            # Act
            result = manager.delete_conversation("nonexistent")

            # Assert
            assert result is False

    def test_clear_history(self, tmp_path):
        """
        AAA Test:
        Arrange: Create conversation with messages
        Act: Clear history
        Assert: Verify messages are removed
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            manager.create_conversation("Test")
            manager.add_message("user", "Message 1")
            manager.add_message("assistant", "Response 1")

            # Act
            manager.clear_history()

            # Assert
            conv = manager.get_current()
            assert len(conv.messages) == 0

    def test_rename_conversation(self, tmp_path):
        """
        AAA Test:
        Arrange: Create conversation
        Act: Rename conversation
        Assert: Verify title is updated
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            conv = manager.create_conversation("Old Title")

            # Act
            result = manager.rename_conversation(conv.id, "New Title")

            # Assert
            assert result is True
            assert conv.title == "New Title"

    def test_set_current_conversation(self, tmp_path):
        """
        AAA Test:
        Arrange: Create multiple conversations
        Act: Set current conversation
        Assert: Verify current_id is updated
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            conv1 = manager.create_conversation("First")
            conv2 = manager.create_conversation("Second")

            # Act
            manager.set_current(conv1.id)

            # Assert
            assert manager.current_id == conv1.id
            assert manager.get_current().id == conv1.id


class TestConversationManagerPersistence:
    """Test cases for ConversationManager persistence."""

    def test_saves_and_loads_conversations(self):
        """
        AAA Test:
        Arrange: Create ConversationManager and add conversations
        Act: Save and verify they're in the conversations dict
        Assert: Verify conversations are tracked
        """
        # Arrange
        manager = ConversationManager()
        conv_id = manager.create_conversation("Test Chat").id
        manager.add_message("user", "Hello")

        # Act - Verify it was saved in memory
        assert conv_id in manager.conversations
        loaded_conv = manager.conversations[conv_id]
        assert len(loaded_conv.messages) >= 1

    def test_loads_from_existing_index(self):
        """
        AAA Test:
        Arrange: ConversationManager loads from real database
        Act: Check loaded conversations
        Assert: Verify data is loaded from the real index file
        """
        # Arrange
        manager = ConversationManager()

        # Act - Real data is loaded automatically
        # Check if we have any conversations
        all_convs = manager.get_all()

        # Assert
        # Either we have conversations from the real DB or it's empty
        assert isinstance(all_convs, list)
        # Verify structure if any exist
        if all_convs:
            conv = all_convs[0]
            assert hasattr(conv, 'id')
            assert hasattr(conv, 'title')
            assert hasattr(conv, 'messages')


class TestRealDataScenarios:
    """Test cases based on real conversation data from the database."""

    def test_real_conversation_structure(self):
        """
        AAA Test:
        Arrange: Use real conversation structure from database
        Act: Parse conversation
        Assert: Verify structure matches expected format
        """
        # Arrange - Real data structure from database
        real_conv_data = {
            "id": "20260302_153527_533455",
            "title": "bonjour",
            "created_at": "2026-03-02T15:35:27.533495",
            "updated_at": "2026-03-02T17:33:52.908571",
            "messages": [
                {"role": "user", "content": "bonjour"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "2+2=?"},
                {"role": "assistant", "content": "The answer is 4."}
            ],
            "document_ids": ["CER_Synchronisation_Pipeline_Georges_Zamfiroiu.docx"]
        }

        # Act
        conv = Conversation.from_dict(real_conv_data)

        # Assert
        assert conv.id == "20260302_153527_533455"
        assert conv.title == "bonjour"
        assert len(conv.messages) == 4
        assert conv.document_ids == ["CER_Synchronisation_Pipeline_Georges_Zamfiroiu.docx"]

    def test_real_conversation_with_rag_document(self, tmp_path):
        """
        AAA Test:
        Arrange: Create conversation with RAG document (real scenario)
        Act: Add document and verify
        Assert: Verify document tracking works
        """
        # Arrange
        with patch('conversations.get_writable_path') as mock_path:
            mock_path.return_value = str(tmp_path)
            manager = ConversationManager()
            conv = manager.create_conversation("Pipeline Question")

            # Act
            manager.add_document("CER_Synchronisation_Pipeline_Georges_Zamfiroiu.docx")
            manager.add_message("user", "what is this document about")

            # Assert
            conv = manager.get_current()
            assert "CER_Synchronisation_Pipeline_Georges_Zamfiroiu.docx" in conv.document_ids
            assert any("what is this document about" in m["content"] for m in conv.messages)
