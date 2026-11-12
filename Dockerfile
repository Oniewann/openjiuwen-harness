# Харнесс кодового агента на базе openJiuwen. Запуск жюри:
#   docker run ... --entrypoint python hack:latest /app/run_agent.py --tasks-dir /data/tasks
FROM python:3.12.14-slim-bookworm

WORKDIR /app

# Зависимости ставятся при сборке; при запуске ничего не скачивается.
# requirements.txt — прямые зависимости, constraints.txt — замок всех транзитивных версий.
COPY requirements.txt constraints.txt /app/
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.txt -c /app/constraints.txt

COPY run_agent.py /app/run_agent.py
COPY harness /app/harness
# Каталог /app при запуске доступен только для чтения: байткод компилируем заранее.
RUN python -m compileall -q /app/harness /app/run_agent.py

# Ничего не писать вне /tmp и не ходить в сеть, кроме прокси моделей.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OTEL_SDK_DISABLED=true

ENTRYPOINT ["python"]
CMD ["/app/run_agent.py", "--tasks-dir", "/data/tasks"]
