FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app
USER app
ENV PATH="/home/app/.local/bin:${PATH}"
WORKDIR /home/app/dashboard

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .
CMD ["voila", "pipe_predictor_demo.ipynb", "--Voila.ip=0.0.0.0", "--port=7860", "--no-browser"]
