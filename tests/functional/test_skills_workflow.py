"""
Functional tests for skills workflow.
Following AAA (Arrange-Act-Assert) pattern.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from skills_manager import SkillsManager, SkillExecutor, SkillInfo


class TestSkillsLoadingWorkflow:
    """Functional tests for loading skills workflow."""

    def test_workflow_load_skills_from_directory(self):
        """
        AAA Test:
        Arrange: Create skills directory with multiple skill files
        Act: Initialize SkillsManager
        Assert: Verify all skills are loaded
        """
        # Arrange
        skills_dir = tempfile.mkdtemp()
        skills_path = Path(skills_dir)

        # Create multiple skill files
        (skills_path / 'skill_pdf_generator.md').write_text('''# PDF Generator

## Description: Generate PDF documents

## Use Cases:
- Creating reports
- Generating invoices

## Instructions:
Use reportlab to create professional PDFs.
''')

        (skills_path / 'skill_data_analyzer.md').write_text('''# Data Analyzer

## Description: Analyze data files

## Instructions:
Use pandas for data analysis.
''')

        config_file = Path(skills_dir) / 'skills_config.json'

        # Act
        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_path
            manager.config_file = config_file
            manager.skills = {}
            manager._load_skills()

            # Assert
            assert 'pdf_generator' in manager.skills
            assert 'data_analyzer' in manager.skills
            assert manager.skills['pdf_generator'].name == 'PDF Generator'
            assert manager.skills['data_analyzer'].name == 'Data Analyzer'

    def test_workflow_loads_enabled_status_from_config(self):
        """
        AAA Test:
        Arrange: Create config with enabled skills
        Act: Load configuration
        Assert: Verify enabled status is restored
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()

        # Create skill files
        (skills_dir / 'skill_first.md').write_text('# First\n## Description: First skill')
        (skills_dir / 'skill_second.md').write_text('# Second\n## Description: Second skill')

        # Create config
        config_file = Path(temp_dir) / 'skills_config.json'
        config_data = {
            'enabled_skills': {
                'first': True,
                'second': False
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Act
        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = config_file
            manager.skills = {
                'first': SkillInfo(
                    id='first', name='First',
                    description='First skill', category='Test', icon='🔧'
                ),
                'second': SkillInfo(
                    id='second', name='Second',
                    description='Second skill', category='Test', icon='🔧'
                )
            }
            manager._load_config()

            # Assert
            assert manager.skills['first'].enabled is True
            assert manager.skills['second'].enabled is False


class TestSkillsCreationWorkflow:
    """Functional tests for skill creation workflow."""

    def test_workflow_create_new_skill(self):
        """
        AAA Test:
        Arrange: Initialize manager
        Act: Create a new skill
        Assert: Verify skill is created and saved
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()
        config_file = Path(temp_dir) / 'skills_config.json'

        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = config_file
            manager.skills = {}

            # Act
            skill_id = manager.create_skill(
                name='Test Skill',
                description='A test skill for automation',
                category='Automation',
                icon='🤖',
                content='Follow these instructions when asked about automation.'
            )

            # Assert
            assert skill_id == 'test_skill'
            assert 'test_skill' in manager.skills
            assert (skills_dir / 'skill_test_skill.md').exists()

    def test_workflow_create_skill_saves_config(self):
        """
        AAA Test:
        Arrange: Create a new skill
        Act: Enable and save
        Assert: Verify config is updated
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()
        config_file = Path(temp_dir) / 'skills_config.json'

        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = config_file
            manager.skills = {}

            # Create skill
            manager.create_skill(
                name='Config Test',
                description='Test',
                category='Test',
                icon='🧪',
                content='Content'
            )

            # Enable skill and save
            manager.skills['config_test'].enabled = True

            # Act
            manager.save_config()

            # Assert
            with open(config_file, 'r') as f:
                config = json.load(f)
            assert 'config_test' in config['enabled_skills']
            assert config['enabled_skills']['config_test'] is True


class TestSkillsApplicationWorkflow:
    """Functional tests for applying skills to conversations."""

    def test_workflow_apply_skills_to_conversation(self):
        """
        AAA Test:
        Arrange: Create manager with enabled skills
        Act: Apply skills to prompt
        Assert: Verify prompt is enhanced
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()

        # Create skill with instructions
        (skills_dir / 'skill_python_helper.md').write_text('''# Python Helper

## Description: Helps with Python code

## Instructions:
When helping with Python code:
1. Follow PEP 8 guidelines
2. Add docstrings to functions
3. Include type hints
4. Provide usage examples
''')

        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = Path(temp_dir) / 'skills_config.json'
            manager.skills = {
                'python_helper': SkillInfo(
                    id='python_helper',
                    name='Python Helper',
                    description='Helps with Python code',
                    category='Development',
                    icon='🐍',
                    enabled=True
                )
            }

            executor = SkillExecutor(manager)
            base_prompt = "You are a helpful assistant."

            # Act
            enhanced_prompt, skill_names = executor.apply_skills_to_prompt(
                message="Help me write Python code",
                base_prompt=base_prompt
            )

            # Assert
            assert 'Python Helper' in enhanced_prompt
            assert 'AVAILABLE SKILLS' in enhanced_prompt
            assert 'PEP 8' in enhanced_prompt
            assert skill_names == ['Python Helper']

    def test_workflow_no_skills_applied_when_none_enabled(self):
        """
        AAA Test:
        Arrange: Create manager with disabled skills
        Act: Apply skills to prompt
        Assert: Verify prompt is unchanged
        """
        # Arrange
        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills = {
                'test': SkillInfo(
                    id='test',
                    name='Test',
                    description='Test',
                    category='Test',
                    icon='🔧',
                    enabled=False
                )
            }

            executor = SkillExecutor(manager)
            base_prompt = "You are helpful."

            # Act
            enhanced_prompt, skill_names = executor.apply_skills_to_prompt(
                message="Test message",
                base_prompt=base_prompt
            )

            # Assert
            assert enhanced_prompt == base_prompt
            assert skill_names == []

    def test_workflow_multiple_skills_applied(self):
        """
        AAA Test:
        Arrange: Create manager with multiple enabled skills
        Act: Apply skills to prompt
        Assert: Verify all skills are included
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()

        (skills_dir / 'skill_first.md').write_text('# First\n## Instructions: First skill instructions')
        (skills_dir / 'skill_second.md').write_text('# Second\n## Instructions: Second skill instructions')
        (skills_dir / 'skill_third.md').write_text('# Third\n## Instructions: Third skill instructions')

        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = Path(temp_dir) / 'skills_config.json'
            manager.skills = {
                'first': SkillInfo(
                    id='first', name='First',
                    description='First', category='Test', icon='1️⃣', enabled=True
                ),
                'second': SkillInfo(
                    id='second', name='Second',
                    description='Second', category='Test', icon='2️⃣', enabled=True
                ),
                'third': SkillInfo(
                    id='third', name='Third',
                    description='Third', category='Test', icon='3️⃣', enabled=False
                )
            }

            executor = SkillExecutor(manager)

            # Act
            enhanced_prompt, skill_names = executor.apply_skills_to_prompt(
                message="Help",
                base_prompt="Base prompt"
            )

            # Assert
            assert 'First skill instructions' in enhanced_prompt
            assert 'Second skill instructions' in enhanced_prompt
            assert 'Third' not in enhanced_prompt
            assert len(skill_names) == 2


class TestSkillsUpdateWorkflow:
    """Functional tests for skill update workflow."""

    def test_workflow_update_existing_skill(self):
        """
        AAA Test:
        Arrange: Create manager with existing skill
        Act: Update skill details
        Assert: Verify skill is updated
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()

        skill_file = skills_dir / 'skill_test.md'
        skill_file.write_text('# Old Name\n## Description: Old description')

        config_file = Path(temp_dir) / 'skills_config.json'

        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = config_file
            manager.skills = {
                'test': SkillInfo(
                    id='test',
                    name='Old Name',
                    description='Old description',
                    category='Old',
                    icon='🔧'
                )
            }

            # Act
            result = manager.update_skill(
                skill_id='test',
                name='New Name',
                description='New description',
                category='New Category',
                icon='🆕',
                content='Updated instructions'
            )

            # Assert
            assert result is True
            assert manager.skills['test'].name == 'New Name'
            assert manager.skills['test'].description == 'New description'
            assert manager.skills['test'].category == 'New Category'
            assert manager.skills['test'].icon == '🆕'

    def test_workflow_update_nonexistent_skill_returns_false(self):
        """
        AAA Test:
        Arrange: Create manager
        Act: Try to update non-existent skill
        Assert: Verify returns False
        """
        # Arrange
        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills = {}

            # Act
            result = manager.update_skill(
                skill_id='nonexistent',
                name='Name',
                description='Desc',
                category='Cat',
                icon='🔧',
                content='Content'
            )

            # Assert
            assert result is False


class TestSkillsDeleteWorkflow:
    """Functional tests for skill deletion workflow."""

    def test_workflow_delete_skill_removes_file(self):
        """
        AAA Test:
        Arrange: Create manager with skill file
        Act: Delete skill
        Assert: Verify file is removed
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()

        skill_file = skills_dir / 'skill_test.md'
        skill_file.write_text('# Test Skill')

        config_file = Path(temp_dir) / 'skills_config.json'

        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = config_file
            manager.skills = {
                'test': SkillInfo(
                    id='test',
                    name='Test',
                    description='Test',
                    category='Test',
                    icon='🔧'
                )
            }

            # Act
            result = manager.delete_skill('test')

            # Assert
            assert result is True
            assert 'test' not in manager.skills
            assert not skill_file.exists()

    def test_workflow_delete_skill_with_image(self):
        """
        AAA Test:
        Arrange: Create skill with associated image
        Act: Delete skill
        Assert: Verify both skill file and image are removed
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()

        skill_file = skills_dir / 'skill_test.md'
        skill_file.write_text('# Test')

        image_file = skills_dir / 'skill_test_icon.png'
        image_file.write_text('fake image')

        config_file = Path(temp_dir) / 'skills_config.json'

        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = config_file
            manager.skills = {
                'test': SkillInfo(
                    id='test',
                    name='Test',
                    description='Test',
                    category='Test',
                    icon='🔧',
                    image_path=str(image_file)
                )
            }

            # Act
            manager.delete_skill('test')

            # Assert
            assert not skill_file.exists()
            assert not image_file.exists()


class TestSkillsCategoryWorkflow:
    """Functional tests for skill categorization."""

    def test_workflow_skills_sorted_by_category(self):
        """
        AAA Test:
        Arrange: Create skills in different categories
        Act: Get all skills
        Assert: Verify skills are grouped by category
        """
        # Arrange
        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills = {
                'pdf_skill': SkillInfo(
                    id='pdf_skill',
                    name='PDF Skill',
                    description='Test',
                    category='Documents',
                    icon='📕'
                ),
                'data_skill': SkillInfo(
                    id='data_skill',
                    name='Data Skill',
                    description='Test',
                    category='AI',
                    icon='🧠'
                ),
                'docx_skill': SkillInfo(
                    id='docx_skill',
                    name='DOCX Skill',
                    description='Test',
                    category='Documents',
                    icon='📘'
                )
            }

            # Act
            skills = manager.get_all_skills()

            # Assert
            # Sorted by category alphabetically (AI < Documents)
            # Then by name within each category
            assert skills[0].category == 'AI'
            assert skills[0].name == 'Data Skill'
            assert skills[1].category == 'Documents'
            assert skills[1].name == 'DOCX Skill'
            assert skills[2].category == 'Documents'
            assert skills[2].name == 'PDF Skill'

    def test_workflow_infers_category_from_skill_id(self):
        """
        AAA Test:
        Arrange: Create skill files with category indicators
        Act: Parse skill files
        Assert: Verify categories are inferred correctly
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        skills_dir = Path(temp_dir) / 'skills'
        skills_dir.mkdir()

        # Create skills with category indicators in filename
        (skills_dir / 'skill_pdf_tool.md').write_text('# PDF Tool')
        (skills_dir / 'skill_rag_search.md').write_text('# RAG Search')
        (skills_dir / 'skill_docx_creator.md').write_text('# DOCX Creator')
        (skills_dir / 'skill_manim_animation.md').write_text('# Manim Animation')

        with patch.object(SkillsManager, '__init__', lambda self: None):
            manager = SkillsManager()
            manager.skills_dir = skills_dir
            manager.config_file = Path(temp_dir) / 'skills_config.json'
            manager.skills = {}

            # Act
            manager._load_skills()

            # Assert
            assert manager.skills['pdf_tool'].category == 'Documents'
            assert manager.skills['rag_search'].category == 'AI'
            assert manager.skills['docx_creator'].category == 'Documents'
            assert manager.skills['manim_animation'].category == 'Visualization'
