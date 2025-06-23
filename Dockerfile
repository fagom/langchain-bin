FROM ollama/ollama

COPY ./run-llama.sh /tmp/run-llama.sh

WORKDIR /tmp

RUN chmod +x run-llama.sh \
    && ./run-llama.sh

EXPOSE 11434