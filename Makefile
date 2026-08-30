PYTHON ?= .venv/bin/python
DEVICE ?= 0
PORT ?= 5999

.DEFAULT_GOAL := help

.PHONY: help run clear test

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

run: ## Start the thermal camera web app (override with DEVICE=, PORT=)
	DEVICE=$(DEVICE) PORT=$(PORT) $(PYTHON) thermal_web.py

clear: ## Reset macOS camera permission cache (tccutil reset Camera)
	tccutil reset Camera

test: ## List AVFoundation devices and test-stream indices 0 and 1 (no file written)
	@echo "AVFoundation video devices:"
	@ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | sed -n '/AVFoundation video devices/,/AVFoundation audio devices/p'
	@echo ""
	@echo "Streaming test - device 1 (2s, no file written):"
	-ffmpeg -f avfoundation -i "1:none" -t 2 -f null -
	@echo ""
	@echo "Streaming test - device 0 (2s, no file written):"
	-ffmpeg -f avfoundation -i "0:none" -t 2 -f null -
