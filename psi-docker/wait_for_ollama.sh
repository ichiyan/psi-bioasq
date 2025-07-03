#!/bin/bash

# start ollama in the background
ollama serve &
# record process id
pid=$!

# pause for ollama to start
sleep 5


# 4.1 GB
echo "🔴 Retrieving Mistral 7B Instruct..."
ollama pull mistral:instruct
echo "🟢 Done pulling Mistral 7B Instruct!"


# wait for ollama process to finish
wait $pid