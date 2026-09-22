CXX = clang++
CXXFLAGS = -O3 -std=c++17 -fPIC -Wall
LDFLAGS = -dynamiclib -framework Metal -framework Foundation
INCLUDES = -Iinclude -Isrc/mzsae/backends/metal/kernels -Isrc/metal

TARGET = libmzsae_metal.dylib
SRCS = src/mzsae/backends/metal/kernels/metal_runtime.mm
KERNEL_SRC = src/mzsae/backends/metal/kernels/mzsae_kernels.metal
PKG_DIR = src/mzsae
BACKEND_DIR = src/mzsae/backends/metal

all: $(TARGET) package_assets

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $< -o $@

package_assets: $(TARGET)
	mkdir -p $(PKG_DIR) $(BACKEND_DIR)
	cp $(TARGET) $(PKG_DIR)/$(TARGET)
	cp $(KERNEL_SRC) $(PKG_DIR)/mzsae_kernels.metal
	cp $(TARGET) $(BACKEND_DIR)/$(TARGET)
	cp $(KERNEL_SRC) $(BACKEND_DIR)/mzsae_kernels.metal

clean:
	rm -f $(TARGET)
	rm -f $(PKG_DIR)/$(TARGET)
	rm -f $(PKG_DIR)/mzsae_kernels.metal
	rm -f $(BACKEND_DIR)/$(TARGET)
	rm -f $(BACKEND_DIR)/mzsae_kernels.metal
	rm -rf build dist src/*.egg-info *.egg-info

format:
	ruff format .

lint:
	ruff check .

PYTHON ?= $(shell if [ -f .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)
PYTEST ?= $(shell if [ -f .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo pytest; fi)

test: all
	$(PYTEST) -v tests/

package: all
	@which uv >/dev/null 2>&1 && uv build || $(PYTHON) -m build

.PHONY: all clean test package_assets format lint package
