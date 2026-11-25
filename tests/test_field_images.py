import base64
import io
import unittest
from unittest.mock import patch
from PIL import Image
from core.perception import VisualPerceptionEngine, UIElement


class FieldImageTests(unittest.TestCase):
    def test_crop_uses_original_pixels_and_is_unannotated(self):
        image = Image.new('RGB', (2000, 1200), 'white')
        element = UIElement(7, 'ComboBoxControl', 'Field', (100,200,500,230), True)
        crops = VisualPerceptionEngine.field_crops(image, [element])
        crop = Image.open(io.BytesIO(base64.b64decode(crops[7])))
        self.assertEqual(crop.size, (424, 54))
        self.assertEqual(crop.getpixel((12,12)), (255,255,255))

    def test_invalid_bounds_and_noneditable_controls_are_excluded(self):
        elements = [UIElement(1,'ButtonControl','Button',(10,10,30,30),True),
                    UIElement(2,'EditControl','Offscreen',(-1,10,30,30),True),
                    UIElement(3,'EditControl','Large',(0,0,1900,1000),True)]
        self.assertEqual(VisualPerceptionEngine.field_crops(Image.new('RGB',(2000,1200)), elements), {})

    def test_screen_capture_occurs_after_tree_scan(self):
        order = []
        engine = VisualPerceptionEngine()
        def tree():
            order.append('tree')
            return [], 'App'
        def screen():
            order.append('screen')
            return Image.new('RGB',(200,100))
        with patch.object(engine,'get_accessibility_tree',side_effect=tree), patch.object(engine,'capture_screen',side_effect=screen):
            engine.perceive()
        self.assertEqual(order, ['tree','screen'])
