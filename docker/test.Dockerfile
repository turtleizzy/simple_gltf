FROM python:3.11-slim
RUN pip install pytest
ENTRYPOINT bash -c "pip install --no-cache-dir --upgrade /app/*.whl && pytest /tests"
