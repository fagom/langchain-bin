#!/usr/bin/env bash

ollama serve &
ollama list
ollama pull llama3.2

ollama serve &
ollama list
ollama pull llama3.2