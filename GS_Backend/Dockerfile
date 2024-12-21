FROM python:3.9

WORKDIR /src

COPY ./src /src
COPY ./data /data

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["fastapi", "run", "src/main.py", "--port", "8000"]
