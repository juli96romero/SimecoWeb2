import json
import time
import logging
import base64
from io import BytesIO
from os import path, listdir

import cv2
import numpy as np
from channels.generic.websocket import WebsocketConsumer
from PIL import Image

from .red import main as load_model_128
from .red_2 import main as load_model_256
from . import views
from . import controller_mov as mov
from .cartesianRemap import apply_fov_remap, fov_optimizer
from .bitstream_optimizer import BitStreamOptimizer
from .CoronalSliceVisualizer import CoronalSliceVisualizer

input_path = "./data/validation/labels"
output_path = "./results/"

# Lazy model loading: they are built on first real use, not at import time,
# so the server startup stays cheap.
_model_128 = None
_model_256 = None


def get_model_128():
    global _model_128
    if _model_128 is None:
        _model_128 = load_model_128("self")
    return _model_128


def get_model_256():
    global _model_256
    if _model_256 is None:
        _model_256 = load_model_256("self")
    return _model_256


bitstream_optimizer = BitStreamOptimizer()


logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("performance.log"),
                       logging.StreamHandler()
                   ])

class MainFrontendConsumer(WebsocketConsumer):
    brightness_levels = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    direction = None
    _last_processing_time = 0
    _total_requests = 0
    _total_processing_time = 0
    meshes = None
    visualizer = None

    def connect(self):
        self.accept()

        if not self.meshes:
            self.meshes = views.get_meshes()

        self.visualizer = CoronalSliceVisualizer(self.meshes, width=300, height=300)
        fov_optimizer.precompute_for_128x128()

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        start_time = time.time()
        self._total_requests += 1

        try:
            # data extraction
            data = json.loads(text_data)
            extract_start = time.time()
            self.direction = data.get('direction')

            # extract brightness levels (overall gain + 8 bands)
            for i in range(9):
                brightness_key = f'brightness{i}' if i > 0 else 'brightness'
                self.brightness_levels[i] = int(data.get(brightness_key, 0))
            extract_time = time.time() - extract_start
            logging.info(f"Parse and extraction time: {extract_time*1000:.2f}ms")

            # transducer movement
            move_time = 0
            current_position = mov.get_current_position()
            current_rotation = mov.get_current_orientation()

            if self.direction == "reset":
                reset_start = time.time()
                current_position = mov.reset_position()
                move_time = time.time() - reset_start
                logging.info(f"Reset time: {move_time*1000:.2f}ms")
            elif self.direction:
                move_start = time.time()
                if data["action"] == "move":
                    current_position = mov.move_transducer(self.direction)
                elif data["action"] == "rotate":
                    current_rotation = mov.rotate_transducer(self.direction)
                move_time = time.time() - move_start
                logging.info(f"Move time: {move_time*1000:.2f}ms")

            # VTK processing (we always render: after moving/resetting the
            # controller we want to see the new slice, and it also ensures
            # clipped is defined for the rest of the pipeline)
            vtk_start = time.time()
            vtk_slice_image, _, _ = self.visualizer.render_from_controller(mov)
            vtk_time = time.time() - vtk_start
            logging.info(f"VTK processing time: {vtk_time*1000:.2f}ms")

            # clip the segmented image
            clipped = None
            if vtk_slice_image is not None:
                clipping_start = time.time()
                clipped, vtk_slice_image = clip_segmented_image(vtk_slice_image)
                clipping_time = time.time() - clipping_start
                logging.info(f"Clipping time: {clipping_time*1000:.2f}ms")
                clipped = cv2.flip(clipped, 0)

            # neural network inference
            if clipped is not None:
                inference_start = time.time()
                inference_image = get_model_128().run_inference(generated_image=clipped)
                inference_time = time.time() - inference_start
                logging.info(f"Inference time: {inference_time*1000:.2f}ms")
            else:
                inference_image = None

            # image post-processing
            image_base64 = None
            fov_image = None
            if inference_image is not None:
                process_start = time.time()

                # brightness adjustment
                brightness_start = time.time()
                brightness_adjusted = apply_banded_brightness(inference_image, self.brightness_levels)
                brightness_time = time.time() - brightness_start
                logging.info(f"  Brightness adjustment: {brightness_time*1000:.2f}ms")

                # apply FOV
                fov_start = time.time()
                fov_image = apply_fov_remap(brightness_adjusted)
                fov_time = time.time() - fov_start
                logging.info(f"  FOV processing: {fov_time*1000:.2f}ms")

                # format as bitstream
                bitstream_start = time.time()
                image_base64 = bitstream_optimizer.formatAsBitStream_optimized(fov_image)
                bitstream_time = time.time() - bitstream_start
                logging.info(f"  Bitstream conversion: {bitstream_time*1000:.2f}ms")

            # total time
            total_time = time.time() - start_time
            self._total_processing_time += total_time

            # performance log
            logging.info(f"TOTAL REQUEST TIME: {total_time*1000:.2f}ms")
            logging.info(f"Average processing time: {self._total_processing_time/self._total_requests*1000:.2f}ms")
            logging.info(f"Estimated FPS: {1/total_time if total_time > 0 else 0:.1f}")
            logging.info("-" * 50)

            # overlay image (segmented clip on top of the ultrasound image)
            overlay_result = None
            if clipped is not None and image_base64 is not None:
                fov_start = time.time()
                overlay = cv2.resize(clipped, (128, 128))
                overlay = apply_fov_remap(overlay)
                overlay_result = cv2.addWeighted(fov_image, 1.0, overlay, 0.5, 0)
                fov_time = time.time() - fov_start
                logging.info(f"  FOV processing overlay: {fov_time*1000:.2f}ms")

            # send response
            response_data = {
                "imageData": image_base64,
                "segmentationImageData": (
                    bitstream_optimizer.formatAsBitStream_optimized(vtk_slice_image)
                    if vtk_slice_image is not None else None
                ),
                "position": current_position,
                "processingTime": total_time,
                "direction": self.direction,
                "rotation": current_rotation,
                "overlayImageData": (
                    bitstream_optimizer.formatAsBitStream_optimized(overlay_result)
                    if overlay_result is not None else None
                )
            }

            self.send(text_data=json.dumps(response_data))

        except Exception as e:
            error_time = time.time() - start_time
            logging.error(f"Error after {error_time*1000:.2f}ms: {str(e)}")
            self.send(text_data=json.dumps({"error": str(e)}))


class ImageConsumer(WebsocketConsumer):

    def connect(self):
        self.accept()

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        image_data = views.vtk_visualization_image(text_data)
        # convert image data to uint8
        image_base64 = formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))


class PickleHandler(WebsocketConsumer):

    index = 0

    def connect(self):
        self.accept()

        # precompute FOV transforms once to improve performance
        fov_optimizer.precompute_for_128x128()

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        input_path = "./results"
        filenames = listdir(input_path)

        if self.index >= len(filenames):
            self.index = 0
        image = cv2.imread(path.join(input_path, filenames[self.index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)

        image_data = apply_fov_remap(image)

        image_base64 = formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))


class Principal128(WebsocketConsumer):  # RV: image-based visualization from the folder
    index = 0

    def connect(self):
        self.accept()

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        input_path = "./data/validation/labels"

        filenames = listdir(input_path)

        image = cv2.imread(path.join(input_path, filenames[self.index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.index += 1
        if self.index >= len(filenames):
            self.index = 0
        image_data = get_model_128().run_inference(generated_image=image)
        image_base64 = formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))


class Principal256(WebsocketConsumer):  # RV: image-based visualization from the folder using the 256px network
    index = 0

    def connect(self):
        self.accept()

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):
        input_path = "./data/validation/labels"

        filenames = listdir(input_path)

        image = cv2.imread(path.join(input_path, filenames[self.index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.index += 1
        if self.index >= len(filenames):
            self.index = 0
        image_data = get_model_256().run_inference(generated_image=image)

        image_base64 = formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))

def formatAsBitStream(image_data):  # RV
    image_data = image_data.astype(np.uint8)

    # reshape the image data to (height, width, channels)
    height, width, channels = image_data.shape
    image_data_reshaped = image_data.reshape((height, width, channels))

    # convert the image array to a PIL Image
    image = Image.fromarray(image_data_reshaped)

    # convert the image to base64
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64

class CombinedSliceConsumer(WebsocketConsumer):

    def connect(self):
        self.accept()

        # precompute FOV transforms once to improve performance
        fov_optimizer.precompute_for_128x128()

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        # generate the image with VTK to use as label
        vtk_slice_image = views.vtk_visualization_image(text_data)

        # send the image to the model so it returns an inference
        image_data = get_model_128().run_inference(generated_image=vtk_slice_image)

        # convert that image to cartesian coordinates
        image_data = apply_fov_remap(image_data)

        image_base64 = formatAsBitStream(image_data=image_data)

        self.send(text_data=json.dumps({"image_data": image_base64}))


class Brightness(WebsocketConsumer):
    index = 0
    brightness_levels = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def connect(self):
        self.accept()

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'You are now connected'
        }))

    def receive(self, text_data):

        input_path = "./data/validation/labels"

        filenames = listdir(input_path)

        image = cv2.imread(path.join(input_path, filenames[self.index]))
        # By default OpenCV uses BGR color space, we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.index += 1
        if self.index >= len(filenames):
            self.index = 0
        image_data = get_model_128().run_inference(generated_image=image)

        try:
            # make sure to convert the value to an integer
            data = json.loads(text_data)
            self.brightness_levels[0] = int(data.get('brightness', 50))
            self.brightness_levels[1] = int(data.get('brightness1', 50))
            self.brightness_levels[2] = int(data.get('brightness2', 50))
            self.brightness_levels[3] = int(data.get('brightness3', 50))
            self.brightness_levels[4] = int(data.get('brightness4', 50))
        except (ValueError, TypeError):
            print("Invalid brightness value. Using the default value")

        brightness_adjusted = apply_banded_brightness(image_data, self.brightness_levels)

        # convert image data to uint8
        image_base64 = formatAsBitStream(image_data=brightness_adjusted)

        self.send(text_data=json.dumps({"image_data": image_base64}))


def apply_banded_brightness(image, brightness_levels):

    # convert the image to int16 to avoid overflow during the operations
    image_int = image.astype(np.int16)

    # apply the overall brightness (gain) adjustment
    adjusted_image = image_int + brightness_levels[0]

    # compute the number of rows and determine the bands
    rows, cols = image.shape[:2]
    band_height = rows // 8  # 8 bands

    # apply the per-band brightness adjustments
    for i in range(8):
        start = i * band_height
        end = (i + 1) * band_height if i < 7 else rows  # the last band reaches the end
        adjusted_image[start:end, :] += brightness_levels[i + 1]  # brightness[1] to brightness[8]

    # clip to keep the values within the range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image

def clip_segmented_image(image):

    img = image.copy().astype(np.float32)
    h, w, _ = img.shape

    # fixed columns (100 to 200)
    col_start, col_end = 100, 200

    y_start = 150  # fixed so it sticks to the skin

    # base mask
    mask = np.ones((h, w), dtype=np.float32) * 0.5

    # enable the detected zone (100x100)
    mask[y_start:y_start+100, col_start:col_end] = 1.0

    # apply the mask
    result = img * mask[:, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)

    # keep the same output as before
    center = image[y_start:y_start+100, col_start:col_end]

    return center, result
