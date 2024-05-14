from fastapi import UploadFile
from .config import router
from pathlib import Path
from web_site.NN.server.src.preprocessing_image import get_formed_clock_image


@router.post("/upload_image_for_preprocessing")
async def upload(image: UploadFile):
    directory = Path("./endpoints/client_files")
    image_path = directory / "image"

    contents = await image.read()
    with open(image_path.absolute(), "wb") as file:
        file.write(contents)

    get_formed_clock_image(image_path)



