from fastapi import UploadFile
from .config import router
from pathlib import Path
from web_site.NN.server.src.time_prediction_neural_network import get_prediction


@router.post("/upload_image_for_NN")
async def upload(image: UploadFile):
    directory = Path("./endpoints/client_files")
    image_path = directory / "image"

    contents = await image.read()
    with open(image_path.absolute(), "wb") as file:
        file.write(contents)

    response = get_prediction(image_path).to_dict()
    return response
