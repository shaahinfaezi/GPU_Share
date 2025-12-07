// SPDX-License-Identifier: MIT
// VRAM Limiter Shim - CUDA Runtime API Version
// This intercepts cudaMalloc/cudaFree which PyTorch actually uses

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

// cudaError_t enum (simplified)
typedef enum {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInsufficientDriver = 35
} cudaError_t;

// Function pointers to real CUDA Runtime functions
static cudaError_t (*real_cudaMalloc)(void** devPtr, size_t size) = NULL;
static cudaError_t (*real_cudaFree)(void* devPtr) = NULL;

// VRAM tracking
static size_t total_allocated_vram = 0;
static size_t vram_limit_mb = 0;
static pthread_mutex_t vram_mutex = PTHREAD_MUTEX_INITIALIZER;
static int shim_initialized = 0;

// Logging
#define LOG_PREFIX "[VRAM_LIMITER_RT] "
#define LOG_FILE "/tmp/vram_limiter.log"

static void log_message(const char* format, ...) {
    FILE* f = fopen(LOG_FILE, "a");
    if (!f) return;
    va_list args;
    va_start(args, format);
    fprintf(f, LOG_PREFIX);
    vfprintf(f, format, args);
    fprintf(f, "\n");
    va_end(args);
    fclose(f);
}

// Shim initialization
static void initialize_shim() {
    if (shim_initialized) return;
    shim_initialized = 1;

    // Clear log file
    FILE* f = fopen(LOG_FILE, "w");
    if (f) fclose(f);

    log_message("Initializing VRAM limiter shim (Runtime API) for PID %d.", getpid());

    // Get VRAM limit from environment variable
    const char* vram_limit_env = getenv("GPU_VRAM_LIMIT_MB");
    if (vram_limit_env) {
        vram_limit_mb = (size_t)atoi(vram_limit_env);
        log_message("VRAM limit set to %zu MB from environment variable.", vram_limit_mb);
    } else {
        log_message("WARNING: GPU_VRAM_LIMIT_MB not set. No VRAM limit will be enforced.");
    }

    // Load real CUDA Runtime functions
    void* cudart_lib = dlopen("libcudart.so", RTLD_LAZY);
    if (!cudart_lib) {
        log_message("FATAL: Failed to open libcudart.so: %s", dlerror());
        return;
    }

    real_cudaMalloc = dlsym(cudart_lib, "cudaMalloc");
    real_cudaFree = dlsym(cudart_lib, "cudaFree");

    if (!real_cudaMalloc) {
        log_message("FATAL: Failed to find cudaMalloc symbol.");
    }
    if (!real_cudaFree) {
        log_message("FATAL: Failed to find cudaFree symbol.");
    }

    log_message("Shim initialized successfully.");
}

// Intercepted cudaMalloc
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!shim_initialized) initialize_shim();
    if (!real_cudaMalloc) {
        log_message("ERROR: real_cudaMalloc not initialized");
        return cudaErrorInsufficientDriver;
    }

    pthread_mutex_lock(&vram_mutex);
    size_t limit_bytes = vram_limit_mb * 1024 * 1024;
    log_message("cudaMalloc request for %zu bytes. Current usage: %zu bytes. Limit: %zu bytes.",
                size, total_allocated_vram, limit_bytes);

    if (vram_limit_mb > 0 && (total_allocated_vram + size) > limit_bytes) {
        pthread_mutex_unlock(&vram_mutex);
        log_message("ERROR: VRAM allocation of %zu bytes denied. Exceeds limit of %zu MB.",
                    size, vram_limit_mb);
        return cudaErrorMemoryAllocation;
    }

    cudaError_t result = real_cudaMalloc(devPtr, size);
    if (result == cudaSuccess) {
        total_allocated_vram += size;
        log_message("Allocation successful. New total usage: %zu bytes (%.2f MB).", 
                    total_allocated_vram, total_allocated_vram / (1024.0 * 1024.0));
    } else {
        log_message("Real cudaMalloc failed with error code: %d", result);
    }
    pthread_mutex_unlock(&vram_mutex);
    return result;
}

// Intercepted cudaFree
cudaError_t cudaFree(void* devPtr) {
    if (!shim_initialized) initialize_shim();
    if (!real_cudaFree) {
        log_message("ERROR: real_cudaFree not initialized");
        return cudaErrorInsufficientDriver;
    }

    // Note: We can't track the freed size accurately without maintaining an allocation map
    log_message("cudaFree called for ptr: %p. VRAM tracking may become inaccurate.", devPtr);

    cudaError_t result = real_cudaFree(devPtr);
    return result;
}
