FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY src ./src
COPY notebooks ./notebooks
COPY data ./data

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt jupyter

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
