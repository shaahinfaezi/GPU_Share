# Makefile for VRAM Limiter Shim

# Compiler and flags
CC = gcc
CFLAGS = -I/usr/local/cuda/include -fPIC -shared -Wl,-soname,libvram-limiter.so
LDFLAGS = -ldl -lpthread

# Target libraries
TARGET_DRIVER = vram_limiter.so
TARGET_RUNTIME = vram_limiter_runtime.so

# Source files
SRC_DRIVER = vram_limiter.c
SRC_RUNTIME = vram_limiter_runtime.c

# Default target - build both
all: $(TARGET_DRIVER) $(TARGET_RUNTIME)

$(TARGET_DRIVER): $(SRC_DRIVER)
	$(CC) $(CFLAGS) -o $(TARGET_DRIVER) $(SRC_DRIVER) $(LDFLAGS)

$(TARGET_RUNTIME): $(SRC_RUNTIME)
	$(CC) $(CFLAGS) -o $(TARGET_RUNTIME) $(SRC_RUNTIME) $(LDFLAGS)

# Installation
install:
	install -m 755 $(TARGET_DRIVER) /usr/local/lib/
	install -m 755 $(TARGET_RUNTIME) /usr/local/lib/

# Clean up
clean:
	rm -f $(TARGET_DRIVER) $(TARGET_RUNTIME)
