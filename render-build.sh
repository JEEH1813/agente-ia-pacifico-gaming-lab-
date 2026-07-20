#!/usr/bin/env bash
# exit on error
set -o errexit

# Actualizar pip e instalar dependencias de Python
pip install --upgrade pip
pip install -r requirements.txt