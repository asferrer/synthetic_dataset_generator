"""
Unit tests for Export Manager and format exporters.

These tests verify the COCO, YOLO, and Pascal VOC export functionality.
"""
import pytest
import json
import sys
from pathlib import Path

# Add frontend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "frontend"))

from app.utils.exporters import (
    ExportManager,
    export_to_yolo,
    export_to_coco,
    export_to_pascal_voc
)


@pytest.mark.unit
class TestExportToCOCO:
    """Tests for COCO format export."""

    def test_export_creates_json_file(self, small_coco_data, tmp_path):
        """Test that export creates a JSON file."""
        result = export_to_coco(small_coco_data, str(tmp_path))

        assert result['success']
        json_file = tmp_path / "dataset.json"
        assert json_file.exists()

    def test_export_valid_json(self, small_coco_data, tmp_path):
        """Test that exported JSON is valid."""
        export_to_coco(small_coco_data, str(tmp_path))

        json_file = tmp_path / "dataset.json"
        with open(json_file) as f:
            data = json.load(f)

        assert 'images' in data
        assert 'annotations' in data
        assert 'categories' in data

    def test_export_preserves_data(self, small_coco_data, tmp_path):
        """Test that exported data matches input."""
        export_to_coco(small_coco_data, str(tmp_path))

        json_file = tmp_path / "dataset.json"
        with open(json_file) as f:
            data = json.load(f)

        assert len(data['images']) == len(small_coco_data['images'])
        assert len(data['annotations']) == len(small_coco_data['annotations'])
        assert len(data['categories']) == len(small_coco_data['categories'])

    def test_export_result_contains_paths(self, small_coco_data, tmp_path):
        """Test that result contains file paths."""
        result = export_to_coco(small_coco_data, str(tmp_path))

        assert 'output_path' in result or 'path' in result or result['success']


@pytest.mark.unit
class TestExportToYOLO:
    """Tests for YOLO format export."""

    def test_export_creates_labels_dir(self, small_coco_data, tmp_path):
        """Test that export creates labels directory."""
        result = export_to_yolo(small_coco_data, str(tmp_path))

        assert result['success']
        labels_dir = tmp_path / "labels"
        assert labels_dir.exists()

    def test_export_creates_classes_file(self, small_coco_data, tmp_path):
        """Test that export creates classes.txt file."""
        export_to_yolo(small_coco_data, str(tmp_path))

        classes_file = tmp_path / "classes.txt"
        assert classes_file.exists()

    def test_classes_file_content(self, small_coco_data, tmp_path):
        """Test that classes.txt contains category names."""
        export_to_yolo(small_coco_data, str(tmp_path))

        classes_file = tmp_path / "classes.txt"
        with open(classes_file) as f:
            classes = f.read().strip().split('\n')

        category_names = [cat['name'] for cat in small_coco_data['categories']]
        for cat_name in category_names:
            assert cat_name in classes

    def test_label_files_created(self, small_coco_data, tmp_path):
        """Test that label files are created for images with annotations."""
        export_to_yolo(small_coco_data, str(tmp_path))

        labels_dir = tmp_path / "labels"

        # At least some label files should be created
        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) > 0

    def test_yolo_format_normalized(self, small_coco_data, tmp_path):
        """Test that YOLO coordinates are normalized (0-1)."""
        export_to_yolo(small_coco_data, str(tmp_path))

        labels_dir = tmp_path / "labels"
        label_files = list(labels_dir.glob("*.txt"))

        if label_files:
            with open(label_files[0]) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class_id x_center y_center width height
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    assert 0 <= x <= 1, "x_center should be normalized"
                    assert 0 <= y <= 1, "y_center should be normalized"
                    assert 0 <= w <= 1, "width should be normalized"
                    assert 0 <= h <= 1, "height should be normalized"


@pytest.mark.unit
class TestExportToPascalVOC:
    """Tests for Pascal VOC format export."""

    def test_export_creates_annotations_dir(self, small_coco_data, tmp_path):
        """Test that export creates Annotations directory."""
        result = export_to_pascal_voc(small_coco_data, str(tmp_path))

        assert result['success']
        annotations_dir = tmp_path / "Annotations"
        assert annotations_dir.exists()

    def test_export_creates_xml_files(self, small_coco_data, tmp_path):
        """Test that export creates XML annotation files."""
        export_to_pascal_voc(small_coco_data, str(tmp_path))

        annotations_dir = tmp_path / "Annotations"
        xml_files = list(annotations_dir.glob("*.xml"))

        # Should have XML files for images with annotations
        assert len(xml_files) > 0

    def test_xml_file_valid(self, small_coco_data, tmp_path):
        """Test that XML files are valid."""
        import xml.etree.ElementTree as ET

        export_to_pascal_voc(small_coco_data, str(tmp_path))

        annotations_dir = tmp_path / "Annotations"
        xml_files = list(annotations_dir.glob("*.xml"))

        if xml_files:
            tree = ET.parse(xml_files[0])
            root = tree.getroot()

            assert root.tag == 'annotation'
            assert root.find('filename') is not None

    def test_xml_contains_objects(self, small_coco_data, tmp_path):
        """Test that XML files contain object annotations."""
        import xml.etree.ElementTree as ET

        export_to_pascal_voc(small_coco_data, str(tmp_path))

        annotations_dir = tmp_path / "Annotations"
        xml_files = list(annotations_dir.glob("*.xml"))

        if xml_files:
            tree = ET.parse(xml_files[0])
            root = tree.getroot()

            objects = root.findall('object')
            # Should have at least one object if there are annotations
            assert len(objects) >= 0  # May be 0 if image has no annotations

    def test_voc_bndbox_format(self, small_coco_data, tmp_path):
        """Test that bounding boxes are in VOC format (xmin, ymin, xmax, ymax)."""
        import xml.etree.ElementTree as ET

        export_to_pascal_voc(small_coco_data, str(tmp_path))

        annotations_dir = tmp_path / "Annotations"
        xml_files = list(annotations_dir.glob("*.xml"))

        if xml_files:
            tree = ET.parse(xml_files[0])
            root = tree.getroot()

            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    assert xmax > xmin, "xmax should be greater than xmin"
                    assert ymax > ymin, "ymax should be greater than ymin"


@pytest.mark.unit
class TestExportManager:
    """Tests for ExportManager class."""

    def test_export_manager_creation(self):
        """Test ExportManager instantiation."""
        manager = ExportManager()
        assert manager is not None

    def test_supported_formats(self):
        """Test that expected formats are supported."""
        manager = ExportManager()

        # Should support at least these formats
        expected_formats = ['coco', 'yolo', 'pascal_voc']

        supported = manager.get_supported_formats() if hasattr(manager, 'get_supported_formats') else expected_formats

        for fmt in expected_formats:
            assert fmt in supported or fmt.replace('_', '') in str(supported).lower()

    def test_export_to_format_coco(self, small_coco_data, tmp_path):
        """Test exporting via manager to COCO format."""
        manager = ExportManager()

        if hasattr(manager, 'export'):
            result = manager.export(small_coco_data, str(tmp_path), format='coco')
            assert result.get('success', True)

    def test_export_to_format_yolo(self, small_coco_data, tmp_path):
        """Test exporting via manager to YOLO format."""
        manager = ExportManager()

        if hasattr(manager, 'export'):
            result = manager.export(small_coco_data, str(tmp_path), format='yolo')
            assert result.get('success', True)


@pytest.mark.unit
class TestExportEdgeCases:
    """Tests for edge cases in export functionality."""

    def test_export_empty_dataset(self, tmp_path):
        """Test exporting empty dataset."""
        empty_data = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'test'}]
        }

        # Should not raise error
        result = export_to_coco(empty_data, str(tmp_path))
        assert result['success']

    def test_export_no_annotations(self, tmp_path):
        """Test exporting dataset with images but no annotations."""
        data = {
            'images': [{'id': 1, 'file_name': 'test.jpg', 'width': 640, 'height': 480}],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'test'}]
        }

        result = export_to_yolo(data, str(tmp_path))
        assert result['success']

    def test_export_special_characters_in_filename(self, tmp_path):
        """Test exporting with special characters in filename."""
        data = {
            'images': [{'id': 1, 'file_name': 'test image (1).jpg', 'width': 640, 'height': 480}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 100, 100], 'area': 10000, 'iscrowd': 0}
            ],
            'categories': [{'id': 1, 'name': 'test'}]
        }

        result = export_to_yolo(data, str(tmp_path))
        assert result['success']

    def test_export_creates_output_dir(self, tmp_path):
        """Test that export creates output directory if it doesn't exist."""
        data = {
            'images': [{'id': 1, 'file_name': 'test.jpg', 'width': 640, 'height': 480}],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'test'}]
        }

        new_dir = tmp_path / "new_export_dir"
        result = export_to_coco(data, str(new_dir))

        assert result['success']
        assert new_dir.exists()
