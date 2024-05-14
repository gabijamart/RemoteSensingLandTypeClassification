import os
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, BBox, CRS
from PIL import Image, ImageEnhance
import numpy as np
import subprocess
import segmentation
import base64

app = Flask(__name__)

dir_name = os.path.dirname(__file__)

config = SHConfig()
config.sh_client_id = "sh-eb88ec93-b4ea-42a9-aed9-49f0f98a2925"
config.sh_client_secret = "4V282kFJIIcx0HRMEbgA6UXpPiSHsoBB"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.save()

print("Client ID:", config.sh_client_id)
print("Client Secret:", config.sh_client_secret)

try:
    token = config.instance_id  
    print("Access token obtained successfully.")
except Exception as e:
    print("Error obtaining access token:", str(e))

def fetch_image(lat_min, lat_max, lon_min, lon_max, date_start, date_end, evalscript):
    bbox = BBox(bbox=[lon_min, lat_min, lon_max, lat_max], crs=CRS.WGS84)
    
    width, height = 2500, 2500
    
    print(f"Using fixed dimensions: width={width}, height={height}")
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=(date_start, date_end),
                other_args={"dataFilter": {"mosaickingOrder": "leastCC"}},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(width, height),
        config=config
    )
    images = request.get_data()
    return np.array(images[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch-images', methods=['POST'])
def fetch_images():
    data = request.json
    lat_min = data.get('lat_min')
    lat_max = data.get('lat_max')
    lon_min = data.get('lon_min')
    lon_max = data.get('lon_max')
    date_start1 = data.get('date_start1')
    date_end1 = data.get('date_end1')
    date_start2 = data.get('date_start2')
    date_end2 = data.get('date_end2')
    
    evalscript_true_color = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04"],
                output: { bands: 3 }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """
    
    try:
        image1 = fetch_image(lat_min, lat_max, lon_min, lon_max, date_start1, date_end1, evalscript_true_color)
        image2 = fetch_image(lat_min, lat_max, lon_min, lon_max, date_start2, date_end2, evalscript_true_color)
        
        timestamp = int(time.time())
        image_name1 = f'true_color_image1_{timestamp}.tiff'
        image_name2 = f'true_color_image2_{timestamp}.tiff'
        image_path1 = os.path.join(dir_name, 'static', image_name1)
        image_path2 = os.path.join(dir_name, 'static', image_name2)

        # Convert the numpy arrays to PIL images
        img1 = Image.fromarray(image1).convert("RGB")
        img2 = Image.fromarray(image2).convert("RGB")

        # Enhance brightness
        enhancer1 = ImageEnhance.Brightness(img1)
        enhanced_img1 = enhancer1.enhance(1.5)  # Increase brightness by 1.5 times

        enhancer2 = ImageEnhance.Brightness(img2)
        enhanced_img2 = enhancer2.enhance(1.5)  # Increase brightness by 1.5 times

        # Save the enhanced images
        enhanced_img1.save(image_path1)
        enhanced_img2.save(image_path2)

        maskpath = segmentation.get_masks(image_path1, image_path2)

        with open(maskpath, "rb") as image_file:
            mask = base64.b64encode(image_file.read())

        mask = mask.decode('utf-8')

        encodedmask = 'data:image/jpg;base64,' + mask

        print(encodedmask)

        if os.name == 'nt':  # for windows
            os.startfile(image_path1)
            os.startfile(image_path2)
        elif os.name == 'posix':  # for mac and linux
            subprocess.call(['open', image_path1])
            subprocess.call(['open', image_path2])

        print(f"Images saved to {image_path1} and {image_path2}")
        return jsonify({"message": "Images fetched successfully", "mask": encodedmask})
    except Exception as e:
        print(f"Error fetching mask: {str(e)}")
        return jsonify({"message": "Error fetching mask", "error": str(e)}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)