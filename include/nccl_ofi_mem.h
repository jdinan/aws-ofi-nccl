/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 */

#include <gdrapi.h>

typedef struct nvshmem_gdr_buf_handle {
	void *ptr;
	size_t length;
	void *base_ptr;
	gdr_info_t info;
	gdr_mh_t mhandle;
} nccl_ofi_gdr_buf_handle_t;

extern gdr_t gdr_desc;

int nccl_ofi_buffer_register(void *addr, size_t length);
int nccl_ofi_buffer_unregister(void *addr);
void nccl_ofi_buffer_unregister_all();
nccl_ofi_gdr_buf_handle_t *nccl_ofi_get_registered_buffer_handle(void *addr, size_t len);
