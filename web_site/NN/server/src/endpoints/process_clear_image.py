from fastapi import UploadFile
from starlette.responses import FileResponse
from .config import router
from pathlib import Path
from web_site.NN.server.src.preprocessing_image import get_formed_clock_image
from web_site.NN.server.src.time_prediction_neural_network import get_prediction


@router.post("/upload_image_for_full")
async def upload(image: UploadFile):
    directory = Path("./endpoints/client_files")
    image_path = directory / "image.jpg"
    output_image_path = directory / "preprocessing_output_image.jpg"

    contents = await image.read()
    with open(image_path.absolute(), "wb") as file:
        file.write(contents)

    get_formed_clock_image(image_path)
    response = get_prediction(output_image_path).to_dict()
    return response


@router.post("/image")
async def upload():
    directory = Path("./endpoints/client_files")
    image_path = directory / "image.jpg"
    response = FileResponse(image_path)
    return response


@router.post("/output_image")
async def upload():
    directory = Path("./endpoints/client_files")
    output_image_path = directory / "preprocessing_output_image.jpg"
    response = FileResponse(output_image_path)
    return response





