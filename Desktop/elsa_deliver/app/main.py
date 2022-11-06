from typing import Optional

import pandas as pd
from fastapi import UploadFile, File, Request, FastAPI, Form
from fastapi.encoders import jsonable_encoder
from inaSpeechSegmenter import Segmenter
from starlette import status
from starlette.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="template")


@app.get('/home', response_class=HTMLResponse)
def write_home(request: Request):
    return templates.TemplateResponse("basic_form.html", {"request": request})


@app.post('/audio')
def submit(
        detect_gender: Optional[bool] = Form(False),
        audio: UploadFile = File(...)
):
    """
    Load audio file and can split analysis:
        - if detect gender selected then the value changes to True
    """
    try:

        if detect_gender:
            seg = Segmenter(detect_gender=False)
            segmentation = seg(audio.filename)
            segmentation_1 = pd.DataFrame.from_records(segmentation, columns=['labels', 'start', 'stop'])
            segmentation_1_json = segmentation_1.to_json(orient='records')

        else:
            seg = Segmenter(detect_gender=True)
            segmentation = seg(audio.filename)
            segmentation_1 = pd.DataFrame.from_records(segmentation, columns=['labels', 'start', 'stop'])
            segmentation_1_json = segmentation_1.to_json(orient='records')

    except Exception as e:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={'message': str(e)})
    else:
        return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(segmentation_1_json))
