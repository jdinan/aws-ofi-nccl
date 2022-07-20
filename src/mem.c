/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include <nccl_ofi.h>
#include <nccl_ofi_log.h>
#include <nccl_ofi_mem.h>
#include <nccl.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gdrapi.h>

#define NCCL_OFI_INITIAL_REG_ARRAY_SIZE 128

static size_t registered_buffer_array_used = 0;
static size_t registered_buffer_array_size = 0;
static nccl_ofi_gdr_buf_handle_t **registered_buffers = NULL;

int nccl_ofi_buffer_register(void *addr, size_t length) {
	nccl_ofi_gdr_buf_handle_t *handle = NULL;
	size_t i;
	int status = 0, ret;
	struct cudaPointerAttributes attr;

	status = cudaPointerGetAttributes(&attr, addr);

	if (status != cudaSuccess) {
		/* clear CUDA error string. */
		cudaGetLastError();
		NCCL_OFI_WARN("Unable to query pointer attributes.");
		status = ncclSystemError;
		goto out_error;
	}

	if (attr.type == cudaMemoryTypeManaged) {
		NCCL_OFI_WARN("Unable to register managed memory (0x%p)", addr);
		status = ncclSystemError;
		goto out_error;
	}
	else if (attr.type == cudaMemoryTypeHost) {
		NCCL_OFI_WARN("Unable to register host memory (0x%p)", addr);
		// FIXME: Better way to deal with this?
		status = 0;
		goto out_error;
	}
	else if (attr.type == cudaMemoryTypeUnregistered) {
		NCCL_OFI_WARN("Unable to register unregistered memory (0x%p)", addr);
		status = 0;
		goto out_error;
	}

	handle = (nccl_ofi_gdr_buf_handle_t *)calloc(1, sizeof(nccl_ofi_gdr_buf_handle_t));
	if (handle == NULL) {
		NCCL_OFI_WARN("Unable to allocate registered buffer handle.");
		status = ncclSystemError;
		goto out_error;
	}

	NCCL_OFI_WARN("Registering (0x%p - 0x%p) length=%zu", addr, (char*)addr + length, length);

	pthread_mutex_lock(&nccl_ofi_lock);

	if (registered_buffer_array_used == registered_buffer_array_size) {
		size_t new_array_size = registered_buffer_array_size ?
			registered_buffer_array_size * 2 :
			NCCL_OFI_INITIAL_REG_ARRAY_SIZE;
		void *new_buf;

		assert(new_array_size < (SIZE_MAX / sizeof(nccl_ofi_gdr_buf_handle_t)));
		new_buf = realloc(registered_buffers, new_array_size * sizeof(nccl_ofi_gdr_buf_handle_t *));
		if (new_buf == NULL) {
			NCCL_OFI_WARN("Unable to resize the registered buffer array.");
			status = ncclSystemError;
			goto out_unlock;
		}
		registered_buffers = (nccl_ofi_gdr_buf_handle_t **)new_buf;
		registered_buffer_array_size = new_array_size;
	}

	for (i = 0; i < registered_buffer_array_used; i++) {
		nccl_ofi_gdr_buf_handle_t *tmp_handle = registered_buffers[i];
		if (addr > tmp_handle->ptr) {
			void *max_addr;
			max_addr = (void *)((char *)tmp_handle->ptr + tmp_handle->length);
			if (addr < max_addr) {
				ERROR_PRINT("Unable to register overlapping memory regions.\n");
				status = NVSHMEMX_ERROR_INVALID_VALUE;
				goto out_unlock;
			}
			continue;
		} else if (addr == tmp_handle->ptr) {
			if (length != tmp_handle->length) {
				NCCL_OFI_WARN("Unable to register overlapping memory regions.");
				status = ncclSystemError;
			} else {
				NCCL_OFI_WARN("Skipping request to register already registered buffer (0x%p, %zu).",
						addr, length);
				// FIXME: Do we need to reference count on the
				// GDRCopy registration since multiple mhandles
				// will use the same GDRCopy registration?
				status = 0;
			}
			goto out_unlock;
			/* addr < tmp_handle->ptr */
		} else {
			break;
		}
	}

	handle->ptr = addr;
	handle->length = length;

	ret = gdr_pin_buffer(gdr_desc, (unsigned long) addr, length, 0, 0, &handle->mhandle);
	if (ret) {
		NCCL_OFI_WARN("Error in gdr_pin_buffer (%d)", ret);
		status = ncclSystemError;
		goto out_unlock;
	}
	ret = gdr_map(gdr_desc, handle->mhandle, &handle->base_ptr, length);
	if (ret) {
		NCCL_OFI_WARN("Error in gdr_map (%d)", ret);
		status = ncclSystemError;
		goto out_unlock;
	}
	ret = gdr_get_info(gdr_desc, handle->mhandle, &handle->info);
	if (ret) {
		NCCL_OFI_WARN("Error in gdr_get_info (%d)", ret);
		status = ncclSystemError;
		goto out_unlock;
	}

	assert(i < registered_buffer_array_size);
	if (i < registered_buffer_array_used) {
		memmove(&registered_buffers[i + 1],
				&registered_buffers[i],
				sizeof(nccl_ofi_gdr_buf_handle_t *) * (registered_buffer_array_used - i));
	}
	registered_buffers[i] = handle;
	registered_buffer_array_used++;

out_unlock:
	pthread_mutex_unlock(&nccl_ofi_lock);

out_error:
	if (status) free(handle);
	return status;
}

int nccl_ofi_buffer_unregister(void *addr) {
	size_t i, ret;
	int status = 0;

	NCCL_OFI_WARN("Unregistering 0x%p", addr);

	pthread_mutex_lock(&nccl_ofi_lock);

	for (i = 0; i < registered_buffer_array_used; i++) {
		nccl_ofi_gdr_buf_handle_t *tmp_handle = registered_buffers[i];
		if (addr > tmp_handle->ptr) {
			continue;
		} else if (addr == tmp_handle->ptr) {
			if ((i + 1) < registered_buffer_array_used) {
				memmove(&registered_buffers[i],
						&registered_buffers[i + 1],
						sizeof(nccl_ofi_gdr_buf_handle_t *) * (registered_buffer_array_used - i));
			}

			ret = gdr_unmap(gdr_desc, tmp_handle->mhandle, tmp_handle->base_ptr, tmp_handle->length);
			if (ret) {
				NCCL_OFI_WARN("Error in gdr_unmap (%d)", ret);
				status = ncclSystemError;
				break;
			}

			ret = gdr_unpin_buffer(gdr_desc, tmp_handle->mhandle);
			if (ret) {
				NCCL_OFI_WARN("Error in gdr_unpin_buffer (%d)", ret);
				status = ncclSystemError;
				break;
			}

			free(tmp_handle);
			registered_buffer_array_used--;
			break;
			/* addr < tmp_handle->ptr*/
		} else {
			NCCL_OFI_WARN("Could not unmap (0x%p)", addr);
			//status = ncclSystemError;
			break;
		}
	}

	pthread_mutex_unlock(&nccl_ofi_lock);

	return status;
}

void nccl_ofi_buffer_unregister_all() {
	int ret;

	pthread_mutex_lock(&nccl_ofi_lock);

	for (size_t i = 0; i < registered_buffer_array_used; i++) {
		nccl_ofi_gdr_buf_handle_t *tmp_handle = registered_buffers[i];

		ret = gdr_unmap(gdr_desc, tmp_handle->mhandle, tmp_handle->base_ptr, tmp_handle->length);
		if (ret) {
			NCCL_OFI_WARN("Error in gdr_unmap (%d)", ret);
		}

		ret = gdr_unpin_buffer(gdr_desc, tmp_handle->mhandle);
		if (ret) {
			NCCL_OFI_WARN("Error in gdr_unpin_buffer (%d)", ret);
		}

		free(registered_buffers[i]);
	}

	registered_buffer_array_used = 0;

	pthread_mutex_unlock(&nccl_ofi_lock);

	return;
}

nccl_ofi_gdr_buf_handle_t *nccl_ofi_get_registered_buffer_handle(void *addr, size_t len) {
	nccl_ofi_gdr_buf_handle_t *tmp_handle;
	size_t min, max, mid;
	nccl_ofi_gdr_buf_handle_t *ret_handle = NULL;

	//NCCL_OFI_WARN("Buffer lookup addr=0x%p len=%zu", addr, len);

	pthread_mutex_lock(&nccl_ofi_lock);

	if (registered_buffer_array_used == 0) {
		goto out;
	}
#if 1
	min = 0;
	max = registered_buffer_array_used;
	do {
		mid = (max - min) / 2 + min;
		/* We have gone past the end of the loop. */
		if (mid >= registered_buffer_array_used) {
			break;
		}
		tmp_handle = registered_buffers[mid];
		if (addr > tmp_handle->ptr) {
			max_addr = (void *)((char *)tmp_handle->ptr + tmp_handle->length);
			max_len = (uint64_t)((char *)max_addr - (char *)addr);
			if (addr < max_addr) {
				if (len > max_len) {
					NCCL_OFI_WARN("Requested range exceeds registered buffer length (0x%p, %zu) > (0x%p, %zu).",
							addr, len, tmp_handle->ptr, tmp_handle->length);
					ret_handle = NULL;
				} else {
					ret_handle = tmp_handle->handle;
				}
				goto out_unlock;
			}
			min = mid + 1;
		} else if (addr == tmp_handle->ptr) {
			if (len > tmp_handle->length) {
				NCCL_OFI_WARN("Requested range exceeds registered buffer length (0x%p, %zu) > (0x%p, %zu).",
						addr, len, tmp_handle->ptr, tmp_handle->length);
				ret_handle = NULL;
			} else {
				ret_handle = tmp_handle->handle;
			}
			goto out_unlock;
		} else {
			if (mid == 0) {
				break;
			}
			max = mid - 1;
		}
	} while (max >= min);
#else
	for (size_t i = 0; i < registered_buffer_array_used; i++) {
		tmp_handle = registered_buffers[i];
		if (addr >= tmp_handle->ptr && (char*)addr+len <= (char*)tmp_handle->ptr + tmp_handle->length) {
			ret_handle = tmp_handle;
			break;
		}
	}
#endif

	// FIXME: Delete this. tmp_handle shouldn't be used here...
	assert(addr >= tmp_handle->ptr && (char*)addr+len <= (char*)tmp_handle->ptr + tmp_handle->length);

out:
	pthread_mutex_unlock(&nccl_ofi_lock);

	if (ret_handle == NULL) {
		NCCL_OFI_WARN("Unable to find a reference to the requested buffer address.");
	}
	return ret_handle;
}
