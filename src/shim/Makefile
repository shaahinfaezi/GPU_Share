# Makefile for VRAM Limiter Shim

# Compiler and flags
CC = gcc
CFLAGS = -I/usr/local/cuda/include -fPIC -shared -Wl,-soname,libvram-limiter.so
LDFLAGS = -ldl -lpthread

# Target library
TARGET = vram_limiter.so

# Source file
SRC = vram_limiter.c

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Installation
install:
	install -m 755 $(TARGET) /usr/local/lib/

# Clean up
clean:
	rm -f $(TARGET)
