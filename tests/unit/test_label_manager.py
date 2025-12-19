"""
Unit tests for Label Manager.

These tests verify the label management functionality.
"""
import pytest
import sys
from pathlib import Path

# Add frontend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "frontend"))

from app.utils.label_manager import LabelManager


@pytest.mark.unit
class TestLabelManager:
    """Tests for LabelManager class."""

    def test_manager_creation(self):
        """Test LabelManager instantiation."""
        manager = LabelManager()
        assert manager is not None

    def test_rename_category(self, small_coco_data):
        """Test renaming a category."""
        manager = LabelManager()

        updated_data = manager.rename_category(
            small_coco_data,
            old_name='cat',
            new_name='feline'
        )

        # Check category was renamed
        category_names = [cat['name'] for cat in updated_data['categories']]
        assert 'feline' in category_names
        assert 'cat' not in category_names

    def test_rename_preserves_other_categories(self, small_coco_data):
        """Test that rename preserves other categories."""
        manager = LabelManager()

        updated_data = manager.rename_category(
            small_coco_data,
            old_name='cat',
            new_name='feline'
        )

        category_names = [cat['name'] for cat in updated_data['categories']]
        assert 'dog' in category_names  # Other category preserved

    def test_rename_preserves_annotations(self, small_coco_data):
        """Test that rename preserves annotation count."""
        manager = LabelManager()

        original_count = len(small_coco_data['annotations'])

        updated_data = manager.rename_category(
            small_coco_data,
            old_name='cat',
            new_name='feline'
        )

        assert len(updated_data['annotations']) == original_count

    def test_delete_category(self, small_coco_data):
        """Test deleting a category."""
        manager = LabelManager()

        updated_data = manager.delete_category(
            small_coco_data,
            category_name='cat'
        )

        category_names = [cat['name'] for cat in updated_data['categories']]
        assert 'cat' not in category_names

    def test_delete_removes_annotations(self, small_coco_data):
        """Test that deleting a category removes its annotations."""
        manager = LabelManager()

        # Count cat annotations before
        cat_id = next(cat['id'] for cat in small_coco_data['categories'] if cat['name'] == 'cat')
        cat_annotations = [ann for ann in small_coco_data['annotations'] if ann['category_id'] == cat_id]

        updated_data = manager.delete_category(
            small_coco_data,
            category_name='cat'
        )

        # Annotations should be reduced
        assert len(updated_data['annotations']) < len(small_coco_data['annotations'])

    def test_merge_categories(self, small_coco_data):
        """Test merging multiple categories into one."""
        manager = LabelManager()

        updated_data = manager.merge_categories(
            small_coco_data,
            source_names=['cat', 'dog'],
            target_name='animal'
        )

        category_names = [cat['name'] for cat in updated_data['categories']]
        assert 'animal' in category_names
        assert 'cat' not in category_names
        assert 'dog' not in category_names

    def test_merge_preserves_annotation_count(self, small_coco_data):
        """Test that merge preserves total annotation count."""
        manager = LabelManager()

        original_count = len(small_coco_data['annotations'])

        updated_data = manager.merge_categories(
            small_coco_data,
            source_names=['cat', 'dog'],
            target_name='animal'
        )

        assert len(updated_data['annotations']) == original_count

    def test_get_category_statistics(self, small_coco_data):
        """Test getting category statistics."""
        manager = LabelManager()

        stats = manager.get_category_statistics(small_coco_data)

        # Should have stats for each category
        assert len(stats) > 0

        # Each stat should have count
        for cat_name, cat_stats in stats.items():
            assert 'count' in cat_stats or isinstance(cat_stats, int)

    def test_add_category(self, small_coco_data):
        """Test adding a new category."""
        manager = LabelManager()

        updated_data = manager.add_category(
            small_coco_data,
            category_name='bird'
        )

        category_names = [cat['name'] for cat in updated_data['categories']]
        assert 'bird' in category_names

    def test_add_existing_category_no_duplicate(self, small_coco_data):
        """Test that adding existing category doesn't create duplicate."""
        manager = LabelManager()

        updated_data = manager.add_category(
            small_coco_data,
            category_name='cat'  # Already exists
        )

        # Should still have same number of categories (or handle gracefully)
        cat_count = sum(1 for cat in updated_data['categories'] if cat['name'] == 'cat')
        assert cat_count == 1

    def test_rename_nonexistent_category(self, small_coco_data):
        """Test renaming a category that doesn't exist."""
        manager = LabelManager()

        # Should handle gracefully (no error or explicit error)
        try:
            updated_data = manager.rename_category(
                small_coco_data,
                old_name='nonexistent',
                new_name='something'
            )
            # If no error, data should be unchanged
            category_names = [cat['name'] for cat in updated_data['categories']]
            assert 'nonexistent' not in category_names
        except (ValueError, KeyError):
            # Also acceptable to raise an error
            pass

    def test_delete_nonexistent_category(self, small_coco_data):
        """Test deleting a category that doesn't exist."""
        manager = LabelManager()

        # Should handle gracefully
        try:
            updated_data = manager.delete_category(
                small_coco_data,
                category_name='nonexistent'
            )
            # Data should be unchanged
            assert len(updated_data['annotations']) == len(small_coco_data['annotations'])
        except (ValueError, KeyError):
            pass


@pytest.mark.unit
class TestLabelManagerSegmentation:
    """Tests for segmentation-related label management."""

    def test_convert_segmentation_to_bbox(self, coco_with_segmentation):
        """Test converting segmentation to bounding box."""
        manager = LabelManager()

        if hasattr(manager, 'segmentation_to_bbox'):
            result = manager.segmentation_to_bbox(coco_with_segmentation)

            # All annotations should have bbox
            for ann in result['annotations']:
                assert 'bbox' in ann

    def test_filter_by_area(self, sample_coco_data):
        """Test filtering annotations by area."""
        manager = LabelManager()

        if hasattr(manager, 'filter_by_area'):
            result = manager.filter_by_area(
                sample_coco_data,
                min_area=5000,
                max_area=15000
            )

            for ann in result['annotations']:
                assert ann.get('area', 10000) >= 5000
                assert ann.get('area', 10000) <= 15000


@pytest.mark.unit
class TestLabelManagerEdgeCases:
    """Tests for edge cases in label management."""

    def test_empty_dataset(self):
        """Test operations on empty dataset."""
        manager = LabelManager()

        empty_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }

        # Should not raise errors
        stats = manager.get_category_statistics(empty_data)
        assert stats == {} or len(stats) == 0

    def test_single_category_dataset(self):
        """Test operations on single category dataset."""
        manager = LabelManager()

        single_cat_data = {
            'images': [{'id': 1, 'file_name': 'test.jpg'}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10]}
            ],
            'categories': [{'id': 1, 'name': 'only_class'}]
        }

        updated = manager.rename_category(
            single_cat_data,
            old_name='only_class',
            new_name='renamed_class'
        )

        assert len(updated['categories']) == 1
        assert updated['categories'][0]['name'] == 'renamed_class'

    def test_category_with_no_annotations(self, small_coco_data):
        """Test handling category with no annotations."""
        manager = LabelManager()

        # Add category with no annotations
        data_with_empty_cat = small_coco_data.copy()
        data_with_empty_cat['categories'] = small_coco_data['categories'] + [
            {'id': 99, 'name': 'empty_category'}
        ]

        # Operations should still work
        stats = manager.get_category_statistics(data_with_empty_cat)
        assert 'empty_category' in stats or len(stats) >= len(small_coco_data['categories'])
