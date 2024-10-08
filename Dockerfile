FROM python:3.9.19

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]