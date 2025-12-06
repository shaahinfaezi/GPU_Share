// SPDX-License-Identifier: MIT
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <cuda.h>
// Undefine macros from cuda.h to avoid redefinition errors
#undef cuMemAlloc
#undef cuMemFree
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

// Function pointers to the real CUDA functions
static CUresult (*real_cuMemAlloc_v2)(CUdeviceptr* dptr, size_t bytesize) = NULL;
static CUresult (*real_cuMemAlloc)(CUdeviceptr* dptr, size_t bytesize) = NULL;
static CUresult (*real_cuMemFree_v2)(CUdeviceptr dptr) = NULL;
static CUresult (*real_cuMemFree)(CUdeviceptr dptr) = NULL;
static CUresult (*real_cuInit)(unsigned int Flags) = NULL;

// VRAM tracking
static size_t total_allocated_vram = 0;
static size_t vram_limit_mb = 0;
static pthread_mutex_t vram_mutex = PTHREAD_MUTEX_INITIALIZER;

// Logging
#define LOG_PREFIX "[VRAM_LIMITER] "
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
    if (real_cuInit) return;

    // Clear log file
    fclose(fopen(LOG_FILE, "w"));

    log_message("Initializing VRAM limiter shim for PID %d.", getpid());

    // Get VRAM limit from environment variable
    const char* vram_limit_env = getenv("GPU_VRAM_LIMIT_MB");
    if (vram_limit_env) {
        vram_limit_mb = (size_t)atoi(vram_limit_env);
        log_message("VRAM limit set to %zu MB from environment variable.", vram_limit_mb);
    } else {
        log_message("WARNING: GPU_VRAM_LIMIT_MB not set. No VRAM limit will be enforced.");
    }

    // Load real CUDA functions
    void* cuda_lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!cuda_lib) {
        log_message("FATAL: Failed to open libcuda.so.1: %s", dlerror());
        return;
    }

    // Use the _v2 versions if available, otherwise fall back
    real_cuMemAlloc_v2 = dlsym(cuda_lib, "cuMemAlloc_v2");
    real_cuMemAlloc = dlsym(cuda_lib, "cuMemAlloc");
    real_cuMemFree_v2 = dlsym(cuda_lib, "cuMemFree_v2");
    real_cuMemFree = dlsym(cuda_lib, "cuMemFree");
    real_cuInit = dlsym(cuda_lib, "cuInit");

    if (!real_cuInit) {
        log_message("FATAL: Failed to find cuInit symbol.");
    }
    if (!(real_cuMemAlloc_v2 || real_cuMemAlloc)) {
        log_message("FATAL: Failed to find cuMemAlloc or cuMemAlloc_v2 symbols.");
    }
    if (!(real_cuMemFree_v2 || real_cuMemFree)) {
        log_message("FATAL: Failed to find cuMemFree or cuMemFree_v2 symbols.");
    }

    log_message("Shim initialized successfully.");
}

// Intercepted cuInit
CUresult cuInit(unsigned int Flags) {
    initialize_shim();
    log_message("cuInit called with flags: %u", Flags);
    return real_cuInit(Flags);
}

// Intercepted cuMemAlloc
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    if (!real_cuMemAlloc_v2) initialize_shim();

    pthread_mutex_lock(&vram_mutex);
    size_t limit_bytes = vram_limit_mb * 1024 * 1024;
    log_message("cuMemAlloc_v2 request for %zu bytes. Current usage: %zu bytes. Limit: %zu bytes.",
                bytesize, total_allocated_vram, limit_bytes);

    if (vram_limit_mb > 0 && (total_allocated_vram + bytesize) > limit_bytes) {
        pthread_mutex_unlock(&vram_mutex);
        log_message("ERROR: VRAM allocation of %zu bytes denied. Exceeds limit of %zu MB.",
                    bytesize, vram_limit_mb);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult result = real_cuMemAlloc_v2(dptr, bytesize);
    if (result == CUDA_SUCCESS) {
        total_allocated_vram += bytesize;
        log_message("Allocation successful. New total usage: %zu bytes.", total_allocated_vram);
    } else {
        log_message("Real cuMemAlloc_v2 failed with error code: %d", result);
    }
    pthread_mutex_unlock(&vram_mutex);
    return result;
}

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    // This is a fallback for older CUDA versions. We prefer _v2.
    if (real_cuMemAlloc_v2) {
        return cuMemAlloc_v2(dptr, bytesize);
    }
    if (!real_cuMemAlloc) initialize_shim();

    pthread_mutex_lock(&vram_mutex);
    size_t limit_bytes = vram_limit_mb * 1024 * 1024;
    log_message("cuMemAlloc (fallback) request for %zu bytes. Current usage: %zu bytes. Limit: %zu bytes.",
                bytesize, total_allocated_vram, limit_bytes);

    if (vram_limit_mb > 0 && (total_allocated_vram + bytesize) > limit_bytes) {
        pthread_mutex_unlock(&vram_mutex);
        log_message("ERROR: VRAM allocation of %zu bytes denied (fallback). Exceeds limit of %zu MB.",
                    bytesize, vram_limit_mb);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult result = real_cuMemAlloc(dptr, bytesize);
    if (result == CUDA_SUCCESS) {
        total_allocated_vram += bytesize;
        log_message("Allocation successful (fallback). New total usage: %zu bytes.", total_allocated_vram);
    } else {
        log_message("Real cuMemAlloc (fallback) failed with error code: %d", result);
    }
    pthread_mutex_unlock(&vram_mutex);
    return result;
}


// Intercepted cuMemFree
CUresult cuMemFree_v2(CUdeviceptr dptr) {
    if (!real_cuMemFree_v2) initialize_shim();

    // Note: We can't know the size of the allocation being freed.
    // This is a known limitation. We will decrement based on an assumption
    // or require a more complex tracking mechanism if this becomes an issue.
    // For now, we log the free operation. A more advanced version would
    // track allocations in a map.
    log_message("cuMemFree_v2 called for ptr: %p. VRAM tracking may become inaccurate.", (void*)dptr);

    CUresult result = real_cuMemFree_v2(dptr);
    // Since we don't know the size, we can't accurately decrement total_allocated_vram.
    // A real-world implementation would need to map dptr to its size.
    return result;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    if (real_cuMemFree_v2) {
        return cuMemFree_v2(dptr);
    }
    if (!real_cuMemFree) initialize_shim();

    log_message("cuMemFree (fallback) called for ptr: %p. VRAM tracking may become inaccurate.", (void*)dptr);
    CUresult result = real_cuMemFree(dptr);
    return result;
}

/*
A more advanced version of this shim would need to solve the cuMemFree size problem.
A common solution is to also intercept cuMemGetInfo to get total/free memory on the device
at the start, and then track the *deltas* caused by our intercepted cuMemAlloc calls.
However, the simplest approach that works for many use cases is to track allocations
in a hash map, where the key is the device pointer and the value is the size.

Example for a more robust cuMemFree:
1. Add a hash map (e.g., using uthash.h or a simple custom implementation).
   `struct allocation { CUdeviceptr ptr; size_t size; ... };`
2. In cuMemAlloc, if successful, add the allocation to the map:
   `add_to_map(dptr, bytesize);`
3. In cuMemFree, look up the pointer in the map to get its size:
   `size_t freed_size = find_in_map(dptr);`
   `if (freed_size > 0) { total_allocated_vram -= freed_size; remove_from_map(dptr); }`
*/
