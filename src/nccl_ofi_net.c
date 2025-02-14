#include "config.h"

/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */
#define _GNU_SOURCE
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <stack.h>
#include <nccl_ofi_param.h>
#include <ctype.h>
#if HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include "tracepoint.h"

#define EFA_PROVIDER_NAME "efa"
#define IS_EFA_PROVIDER(NAME) (strcmp((NAME), EFA_PROVIDER_NAME)==0)

/* NICs info list for a provider */
struct fi_info* ofi_info_list = NULL;
/* Number of NICs */
int ofi_ndevices = -1;
/*
 * NCCL OFI component array for all NICs. To avoid contention
 * over the libfabric resources that are associated with every
 * component, the netdev to component lookup structure is per
 * thread. So every service thread maintains an array of
 * component structures and these structures/resources are then
 * used by the corresponding proxy thread.
 */
__thread nccl_ofi_t **nccl_ofi_component = NULL;
/* Indicates if memory registration of local buffers is required */
bool local_mr = false;
/* Indicates if remote virtual addressing is used */
bool virt_addr_mr = false;
/* Indicates if memory registration of device buffers is required */
bool hmem_mr = false;
/* Indicates if endpoint memory registration is required */
bool endpoint_mr = false;
/* Indicates if the provider selects MR keys */
bool prov_key_mr = false;
/* Indicates if GPUDirect is supported by libfabric provider */
bool support_gdr = true;
/* Indicates if the cudaDeviceFlushGPUDirectRDMAWrites function should be used
 * to flush data to the GPU. Note, CUDA flush support is not supported on all
 * platforms and should be disabled by default */
bool cuda_flush = false;

/* number of duplicate providers to create for each discovered
 * provider, including renaming to cause NCCL to create additional
 * rings to use the connections
 */
int nic_dup_conns = 0;

/* number of cq entries to read in a single call to fi_cq_read.
   This variable will be updated during init (hence, can not be
   const), but will not change during execution.  Therefore, it may be
   read in the polling loop without protection of a lock. */
static size_t cq_read_count = 1;

// NCCL OFI lock for concurrency
pthread_mutex_t nccl_ofi_lock = PTHREAD_MUTEX_INITIALIZER;
// Logger Function
ncclDebugLogger_t ofi_log_function = NULL;
/*
 * Maximum numbers of requests supported by plugin. Since NCCL Net v5,
 * one NCCL request can correspond to multiple network requests with `n`
 * identifier passed to irecv(). Therefore, the total number of requests
 * that plugin should support is product of number of NCCL requests and
 * maximum number of recvs supported by plugin.
 */
int max_requests = NCCL_OFI_MAX_REQUESTS * NCCL_OFI_MAX_RECVS;

const char *provider_filter = NULL;

/* Table indicating allocation state of MR keys */
static size_t num_mr_keys = 0;
static bool *mr_keys = NULL;

/*
 * @brief	Allocate a memory registration key
 */
static uint64_t allocate_mr_key(int dev)
{
	uint64_t key = FI_KEY_NOTAVAIL;

	if (prov_key_mr) {
		NCCL_OFI_WARN("Invalid call to allocate_mr_key");
		return FI_KEY_NOTAVAIL;
	}

	pthread_mutex_lock(&nccl_ofi_lock);

	for (size_t i = 0; i < num_mr_keys; i++) {
		if (mr_keys[dev * num_mr_keys + i]) {
			mr_keys[dev * num_mr_keys + i] = false;
			key = i;
			break;
		}
	}

	if (key == FI_KEY_NOTAVAIL)
		NCCL_OFI_WARN("No MR keys available (max: %d)", num_mr_keys);

	pthread_mutex_unlock(&nccl_ofi_lock);
	return key;
}

/*
 * @brief	Free a memory registration key
 */
static ncclResult_t free_mr_key(int dev, uint64_t key)
{
	if (prov_key_mr) {
		NCCL_OFI_WARN("Invalid call to free_mr_key");
		return ncclInternalError;
	}

	if (key >= num_mr_keys) {
		NCCL_OFI_WARN("Key value out of range (%"PRIu64")", key);
		return ncclInternalError;
	}

	if (mr_keys[dev * num_mr_keys + key] != false) {
		NCCL_OFI_WARN("Attempted to free a key that's not in use (%"PRIu64")", key);
		return ncclInternalError;
	}

	pthread_mutex_lock(&nccl_ofi_lock);

	mr_keys[dev * num_mr_keys + key] = true;

	pthread_mutex_unlock(&nccl_ofi_lock);

	return ncclSuccess;
}


/*
 * @brief	Allocates free list for NCCL OFI requests
 */
static ncclResult_t allocate_ofi_fl(free_list_t **nccl_ofi_req_fl, size_t fl_size,
				    size_t buffer_size)
{
	ncclResult_t ret = ncclSuccess, idx;
	free_list_t *fl = NULL;
	size_t alloc_size = sizeof(free_list_t) + fl_size * buffer_size;

	/* Validate free list size and buffer size */
	if (fl_size < 1 || buffer_size < 1) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid free list size and/or buffer size. Provided fl_size: %zu and buffer size: %zu",
			       fl_size, buffer_size);
		goto error;
	}

	/* Allocate free list structure */
	fl = (free_list_t *)malloc(alloc_size);
	if (fl == NULL) {
		NCCL_OFI_WARN("Unable to allocate free list");
		ret = ncclSystemError;
		goto error;
	}
	memset(fl, 0, alloc_size);

	fl->size = fl_size;

	/* Allocate stack of free indexes */
	fl->free_index = allocate_stack(fl->size);
	if (fl->free_index == NULL) {
		NCCL_OFI_WARN("Couldn't allocate free index stack");
		ret = ncclSystemError;
		goto error;
	}

	/* Initialise stack */
	for (idx = 0; idx < fl->free_index->size; idx++) {
		ret = stack_push(fl->free_index, idx);
		if (ret != 0)
			goto error;
	}

	*nccl_ofi_req_fl = fl;

	goto exit;

error:
	if (fl->free_index)
		free_stack(fl->free_index);
	if (fl)
		free(fl);
exit:
	return ret;
}

/*
 * @brief	Release free list for NCCL OFI requests
 */
void free_ofi_fl(free_list_t *nccl_ofi_req_fl)
{
	if (!nccl_ofi_req_fl)
		return;

	if (nccl_ofi_req_fl->free_index)
		free_stack(nccl_ofi_req_fl->free_index);

	free(nccl_ofi_req_fl);
}

static const char *nccl_ofi_req_state_str(nccl_ofi_req_state_t state)
{
	switch(state) {
	case NCCL_OFI_REQ_CREATED:
		return "CREATED";
	case NCCL_OFI_REQ_PENDING:
		return "PENDING";
	case NCCL_OFI_REQ_COMPLETED:
		return "COMPLETED";
	case NCCL_OFI_REQ_ERROR:
		return "ERROR";
	default:
		return "unknown";
	}
}

static const char *nccl_ofi_req_direction_str(nccl_ofi_req_direction_t direction)
{
	switch(direction) {
	case NCCL_OFI_SEND:
		return "SEND";
	case NCCL_OFI_RECV:
		return "RECV";
	default:
		return "unknown";
	}
}

/*
 * @brief	Print NCCL OFI request information
 */
static const char *nccl_ofi_request_str(nccl_ofi_req_t *req)
{
	static char buf[256];
	snprintf(buf, sizeof(buf), "{ buffer_index: %lu, dev: %d, size: %zu, state: %s, direction: %s }",
		req->buffer_index,
		req->dev,
		req->size,
		nccl_ofi_req_state_str(req->state),
		nccl_ofi_req_direction_str(req->direction)
	);
	return buf;
}

/*
 * @brief	Assign an allocated NCCL OFI request buffer
 */
static inline nccl_ofi_req_t *allocate_nccl_ofi_request(free_list_t *fl)
{
	nccl_ofi_req_t *req = NULL;
	uint64_t next_avail_index;

	if (OFI_UNLIKELY(fl == NULL || fl->free_index == NULL)) {
		NCCL_OFI_WARN("Free list is empty or Free Index stack does not exist.");
		goto exit;
	}

	/* Get free index */
	next_avail_index = stack_pop(fl->free_index);
	if (OFI_UNLIKELY(next_avail_index >= fl->free_index->size)) {
		NCCL_OFI_WARN("No pre-allocated buffer is available for use. next_avail_index: %lu and free_index Size: %d",
			       next_avail_index, fl->free_index->size);
		goto exit;
	}

	/* Get buffer */
	if (OFI_UNLIKELY(fl->buffers == NULL)) {
		NCCL_OFI_WARN("No pre-allocated buffers are present.");
		goto exit;
	}

	req = &((nccl_ofi_req_t *)fl->buffers)[next_avail_index];
	req->buffer_index = next_avail_index;

exit:
	return req;
}

/*
 * @brief	Zero out NCCL OFI request
 */
static inline void zero_nccl_ofi_req(nccl_ofi_req_t *req)
{
	req->lComm = NULL;
	req->sComm = NULL;
	req->rComm = NULL;

	req->buffer_index = 0ULL;
	memset(&req->ctx, 0, sizeof(struct fi_context));

	req->dev = -1;
	req->size = 0;

	req->state = NCCL_OFI_REQ_CREATED;

	req->direction = -1;
}

/*
 * @brief	Prepares NCCL OFI request for reuse
 */
static inline int free_nccl_ofi_req(nccl_ofi_req_t *req, bool dec_inflight_cmds)
{
	int ret = ncclSuccess;
	sendComm_t *sComm = NULL;
	recvComm_t *rComm = NULL;
	uint64_t buffer_index;

	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Provided null request for cleanup");
		goto exit;
	}

	if (req->direction == NCCL_OFI_SEND) {
		sComm = req->sComm;
		if (OFI_UNLIKELY(sComm == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Invalid sComm provided for request of device %d",
				      sComm->dev);
			goto exit;
		}

		/* Update free list */
		if (OFI_UNLIKELY(sComm->nccl_ofi_reqs_fl == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("sComm for device %d does not have valid free list",
				      sComm->dev);
			goto exit;
		}

		buffer_index = req->buffer_index;

		/* Zero out buffer */
		zero_nccl_ofi_req(req);

		ret = stack_push(sComm->nccl_ofi_reqs_fl->free_index,
				 buffer_index);
		if (OFI_UNLIKELY(ret != 0))
			goto exit;

		/* Reduce inflight commands */
		if (OFI_LIKELY(dec_inflight_cmds == true))
			sComm->num_inflight_reqs--;

	}
	else if (req->direction == NCCL_OFI_RECV) {
		rComm = req->rComm;
		if (OFI_UNLIKELY(rComm == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Invalid rComm provided for request of device %d",
				      rComm->dev);
			goto exit;
		}

		/* Update free list */
		if (OFI_UNLIKELY(rComm->nccl_ofi_reqs_fl == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("rComm for device %d does not have valid free list",
				      rComm->dev);
			goto exit;
		}

		buffer_index = req->buffer_index;

		/* Zero out buffer */
		zero_nccl_ofi_req(req);

		ret = stack_push(rComm->nccl_ofi_reqs_fl->free_index,
				 buffer_index);
		if (OFI_UNLIKELY(ret != 0))
			goto exit;

		/* Reduce inflight commands */
		if (OFI_LIKELY(dec_inflight_cmds == true))
			rComm->num_inflight_reqs--;
	}
	else {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unexpected transaction direction. Transaction direction: %d",
			       req->direction);
	}

exit:
	return ret;
}

static int in_list(const char *item, const char *list)
{
	int ret = 0;
	char *token = NULL;
	char *list_temp = strdup(list);

	if (list_temp == NULL) {
		if (list != NULL) {
			NCCL_OFI_WARN("Unable to duplicate list.");
			ret = ncclSystemError;
		}
		goto exit;
	}

	token = strtok((char *)list_temp, ",");

	while (token) {
		if (strcmp(item, token) == 0) {
			ret = 1;
			goto exit;
		}
		token = strtok(NULL, ",");
	}

exit:
	free(list_temp);
	return ret;
}

/*
 * @brief	Returns true if the given provider matches IPv6 addressing format,
 *		interfaces from tcp_if_exclude_list or multiple memory tag formats.
 *
 * @return 	true, if success
 *		false, otherwise
 */
static bool match_prov_info(char *name, uint32_t addr_format,
			    uint64_t mem_tag_format, uint64_t expected_mem_tag_format)
{
	const char *tcp_if_exclude_list = ofi_nccl_exclude_tcp_if();

	if (in_list(name, tcp_if_exclude_list)) {
		return true;
	} else if (!ofi_nccl_use_ipv6_tcp() && (addr_format == FI_SOCKADDR_IN6)) {
		return true;
	} else if (mem_tag_format != expected_mem_tag_format) {
		/* TODO: Remove after https://github.com/ofiwg/libfabric/issues/6126 is fixed */
		/* RxM utility provider adds `FI_COLLECTIVE` capability
		 * which ends up duplicating the fi_info structures. That
		 * is because the size of the supported tag changes when
		 * `FI_COLLECTIVE` is enabled.
		 * This happens even when applications do not request for
		 * this capability in hints.
		 * For now, we choose one tag format and use that to filter all
		 * info objects.
		 */
		return true;
	}

	return false;
}

/*
 * @brief	Removes info objects from global `ofi_info_list` matching
 *		certain criteria for TCP provider.
 */
static void filter_tcp_info_list()
{
	struct fi_info *prev = NULL, *curr = NULL;
	struct fi_info *delete_info = NULL;
	bool delete_prov = false;
	uint64_t expected_mem_tag_format = 0;

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Removing unnecessary interfaces and address formats for TCP provider");

	curr = ofi_info_list;
	expected_mem_tag_format = curr->ep_attr->mem_tag_format;

	while (curr != NULL) {

		/* Check if interface name and format matches deletion criteria */
		delete_prov = match_prov_info(curr->domain_attr->name,
					      curr->addr_format,
					      curr->ep_attr->mem_tag_format,
					      expected_mem_tag_format);
		if (delete_prov) {

			if (prev != NULL) {
				prev->next = curr->next;
			}
			ofi_ndevices--;

			delete_info = curr;
			curr = curr->next;

			/* Delete node matching criteria */
			delete_info->next = NULL;
			fi_freeinfo(delete_info);
		}
		else {
			if (prev == NULL) {
				/*
				 * Update HEAD of ofi_info_list to point to first endpoint which
				 * can be used for communication.
				 */
				ofi_info_list = curr;
			}

			prev = curr;
			curr = curr->next;
		}
	}

	/*
	 * In case all info objects match the filter criteria,
	 * update HEAD of ofi_info_list to point to NULL.
	 */
	if (prev == NULL) {
		ofi_info_list = prev;
	}
}

#if HAVE_CUDA
/*
 * @brief	Gets the CUDA device associated with the buffer
 *
 * @param	data
 *		Pointer to CUDA buffer.
 *
 * @return	Valid CUDA device ID on success
 *		-1 on error
 * @return	0 on success
 *		non-zero on error
 */
static ncclResult_t get_cuda_device(void *data, int *device)
{
	ncclResult_t ret = ncclSuccess;
	int cuda_device = -1;
	struct cudaPointerAttributes attr;
	cudaError_t cuda_ret = cudaPointerGetAttributes(&attr, data);

	if (cuda_ret != cudaSuccess) {
		ret = ncclUnhandledCudaError;
		NCCL_OFI_WARN("Invalid buffer pointer provided");
		goto exit;
	}

	if (attr.type == cudaMemoryTypeDevice) {
		cuda_device = attr.device;
	}
	else {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid type of buffer provided. Only device memory is expected for NCCL_PTR_CUDA type");
	}

exit:
	*device = cuda_device;
	return ret;
}
#endif

/*
 * @brief	Registers memory region (both HOST and CUDA)
 *
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static ncclResult_t register_mr_buffers(ofiComm_t *comm, void *data,
					size_t size, int type,
					struct fid_mr **mr_handle)
{
	ncclResult_t ret = ncclSuccess;
	int rc;
	struct fi_mr_attr mr_attr = {0};
	struct iovec iov = {0};

	/* Check if provider requires registration of local buffers */
	if ((local_mr != true) && (type == NCCL_PTR_HOST)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			"Skip registering host buffer. local_mr: %d", local_mr);
		goto exit;
	}

	/* Populate IOV vector for memory registration */
	iov.iov_base = data;
	iov.iov_len = size;

	/* Initialize MR attributes */
	mr_attr.mr_iov = &iov;
	mr_attr.iov_count = 1;
	mr_attr.access = FI_SEND | FI_RECV;

	switch (type) {
	case NCCL_PTR_HOST:
		mr_attr.access |= FI_READ;
		mr_attr.iface = FI_HMEM_SYSTEM;
		break;
#if HAVE_CUDA
	case NCCL_PTR_CUDA:
		mr_attr.access |= FI_REMOTE_READ;
		mr_attr.iface = FI_HMEM_CUDA;

		/* Get CUDA device ID */
		ret = get_cuda_device(data, &mr_attr.device.cuda);
		if (OFI_UNLIKELY(ret != ncclSuccess)) {
			goto exit;
		}
		break;
#endif
#if HAVE_NEURON
	case NCCL_PTR_NEURON:
		mr_attr.access |= FI_REMOTE_READ;
		mr_attr.iface = FI_HMEM_NEURON;
		/*
		 * Store a sentinel; libfabric requires this to be initialized Libfabric
		 * requires the device.neuron field to be set for Neuron HMEM, but the EFA
		 * provider does not use the value.  Store an invalid device id sentinel to
		 * both follow the Libfabric spec and cause an error if a provider uses the
		 * value in the future.
		 */
		mr_attr.device.neuron = -1;
		break;
#endif
	default:
		ret = ncclInternalError;
		goto exit;
	}

	if (!prov_key_mr) {
		uint64_t key = allocate_mr_key(comm->dev);
		if (key == FI_KEY_NOTAVAIL) {
			NCCL_OFI_WARN("MR key allocation failed");
			ret = ncclSystemError;
			goto exit;
		}
		mr_attr.requested_key = key;
	}

	rc = fi_mr_regattr(comm->baseComm.ofi_comp->domain,
			    &mr_attr, 0, mr_handle);
	if (OFI_UNLIKELY(rc != 0)) {
		NCCL_OFI_WARN("Unable to register memory (type = %d) for device %d. RC: %d, Error: %s",
			       type, comm->dev, rc, fi_strerror(-rc));
		ret = ncclSystemError;
		goto exit;
	}

	if (endpoint_mr) {
		rc = fi_mr_bind(*mr_handle, (fid_t)nccl_ofi_component[comm->dev]->ep, 0);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Unable to bind MR to EP (type = %d) for device %d. RC: %d, Error: %s",
				       type, comm->dev, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto exit;
		}

		rc = fi_mr_enable(*mr_handle);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Unable to enable MR (type = %d) for device %d. RC: %d, Error: %s",
				       type, comm->dev, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto exit;
		}
	}

exit:
	return ret;
}

/*
 * @brief	Returns hints info structure depending on GPUDirect support requirement
 */
static void get_hints(struct fi_info *hints, int request_gdr)
{
	if (request_gdr) {
		hints->caps = FI_TAGGED | FI_MSG | FI_HMEM | FI_REMOTE_COMM;
		if (!cuda_flush)
			hints->caps |= FI_RMA | FI_READ;
		/*
		 * Set MR mode bits to indicate that application allows
		 * registration of both local and device memory buffers
		 * and can support the endpoint memory registration model
		 */
		hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_ENDPOINT;
		hints->domain_attr->mr_key_size = (size_t) ofi_nccl_mr_key_size();
	}
	else {
		hints->caps = FI_TAGGED | FI_MSG | FI_REMOTE_COMM;
		/*
		 * Set MR mode bits to indicate that application allows
		 * registration of both local memory buffers
		 */
		hints->domain_attr->mr_mode = FI_MR_LOCAL;
	}

	hints->mode = FI_CONTEXT;

	hints->ep_attr->type = FI_EP_RDM;

	hints->domain_attr->threading = FI_THREAD_SAFE;

	/* Set progress mode to unspec to use the provider's default mode. */
	hints->domain_attr->control_progress = FI_PROGRESS_UNSPEC;
	hints->domain_attr->data_progress = FI_PROGRESS_UNSPEC;

	/* Set MR mode bits to indicate FI_MR_BASIC registration */
	hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;

	hints->tx_attr->msg_order = FI_ORDER_SAS;
	hints->rx_attr->msg_order = FI_ORDER_SAS;
}

/*
 * @brief	Returns provider info structure. It first tries to get providers
 *		which supports GPUDirect. If not found, it re-tries to search for
 *		provider supporting tagged messaging and RDM endpoints.
 */
static int find_ofi_provider(struct fi_info **providers)
{
	int rc = 0;
	struct fi_info *gdr_hints, *hints;

	gdr_hints = fi_allocinfo();
	hints = fi_allocinfo();
	if ((gdr_hints == NULL) || (hints == NULL)) {
		NCCL_OFI_WARN("Unable to allocate hints fi_info structure");
		rc = -FI_ENOMEM;
		goto exit;
	}

	/* Get hints for GPUDirect capable provider */
	get_hints(gdr_hints, true);

	rc = fi_getinfo(ofi_version, NULL, NULL, 0ULL, gdr_hints, providers);
	if (rc == -FI_ENODATA) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Could not find any optimal provider supporting GPUDirect RDMA");

		/* Indicate that plugin doesn't support transfers using GPU buffers */
		support_gdr = false;
#if !HAVE_NEURON
		/* Functioning without GDR support is not a valid use case for neuron */
		/* Re-try finding non-GPUDirect capable provider */
		get_hints(hints, false);

		rc = fi_getinfo(ofi_version, NULL, NULL, 0ULL, hints, providers);
		if (rc == -FI_ENODATA) {
			NCCL_OFI_WARN("Couldn't find any optimal provider");
		} else if (rc != 0) {
			NCCL_OFI_WARN("OFI call failed with RC %d, %s", rc, fi_strerror(-rc));
		}
#endif
	}
	else if (rc != 0) {
		NCCL_OFI_WARN("OFI call failed with RC %d, %s", rc, fi_strerror(-rc));
	}

exit:
	if (gdr_hints)
		fi_freeinfo(gdr_hints);
	if (hints)
		fi_freeinfo(hints);
	return rc;
}

/*
 * @brief	Calls fi_getinfo() to find a list of usable providers for RDM
 *		tagged endpoints.  The code will return all providers
 *		with the same name as the first provider in the list
 *		returned by fi_getinfo() that satisfies any filters
 *		applied by prov_include.
 *
 * @param	prov_include
 *		Contains a list of preferred provider names.
 *
 * @return	A list of fi_info structures for a single provider.
 * @return	0 on success
 *		non-zero on error
 */
static int get_ofi_provider(const char *prov_include, struct fi_info **prov_info_list)
{
	int rc = 0;
	struct fi_info *providers = NULL, *prov = NULL, *last_prov;
	char *selected_prov_name = NULL;

	rc = find_ofi_provider(&providers);
	if (rc != 0)
		goto error;

	if (!providers)
		goto error;

	/* Pick a provider name to use.  If there is a prov_include
	 * provided, use the first provider which matches the list,
	 * otherwise use the first provider in the list.
	 */
	if (prov_include) {
		prov = providers;
		while (prov) {
			if (in_list(prov->fabric_attr->prov_name, prov_include)) {
				selected_prov_name = prov->fabric_attr->prov_name;
				break;
			}
			prov = prov->next;
		}
	} else {
		selected_prov_name = providers->fabric_attr->prov_name;
	}
	if (!selected_prov_name)
		goto error;

	/* Now remove all providers in the providers list that do not
	 * match the selected name, and count the ones that do.
	 */
	prov = providers;
	providers = NULL;
	last_prov = NULL;
	ofi_ndevices = 0;
	while (prov) {
		struct fi_info *prov_next = prov->next;
		prov->next = NULL;

		if (strcmp(selected_prov_name, prov->fabric_attr->prov_name) != 0) {
			fi_freeinfo(prov);
		} else {
			if (!providers) {
				providers = last_prov = prov;
			} else {
				last_prov->next = prov;
				last_prov = prov;
			}
			ofi_ndevices++;
		}
		prov = prov_next;
	}

	*prov_info_list = providers;
	if (ofi_ndevices == 0)
		goto error;

	return ncclSuccess;

error:
	if (providers)
		fi_freeinfo(providers);
	return ncclSystemError;
}

/*
 * @brief	Returns provider info structure for the given NIC ID.
 */
static struct fi_info *get_nic_info(int dev, struct fi_info *nic_info_list)
{
	int dev_idx = 0;
	struct fi_info *nic_info = NULL;

	nic_info = nic_info_list;
	while ((nic_info != NULL) && (dev_idx < dev)) {
		dev_idx++;
		nic_info = nic_info->next;
	}

	return nic_info;
}

/*
 * @brief	Allocates and initialises various libfabric resources like
 *		fabric, domain, endpoint, CQ and AV.
 *
 * @return	Initialised nccl_ofi_comp structure
 */
static ncclResult_t create_nccl_ofi_component(struct fi_info *prov,
				     nccl_ofi_t *nccl_ofi_comp)
{
	ncclResult_t ret = ncclSuccess;
	struct fi_cq_attr cq_attr = {0};
	struct fi_av_attr av_attr = {0};
	int ofi_tag_leading_zeroes = 0, ofi_tag_bits_for_ring_id = 64;

	/* Determine if any tag bits are used by provider */
	while (!((prov->ep_attr->mem_tag_format << ofi_tag_leading_zeroes++) &
		(uint64_t) OFI_HIGHEST_TAG_BIT) &&
		(ofi_tag_bits_for_ring_id >= MIN_TAG_BITS_FOR_RING_ID)) {
		ofi_tag_bits_for_ring_id--;
	}

	if (OFI_UNLIKELY(ofi_tag_bits_for_ring_id < MIN_TAG_BITS_FOR_RING_ID)) {
		NCCL_OFI_WARN("Provider %s does not provide enough tag bits %d for ring ID. Minimum required is %d",
			      prov->fabric_attr->prov_name,
			      ofi_tag_bits_for_ring_id,
			      MIN_TAG_BITS_FOR_RING_ID);
		ret = ncclSystemError;
		goto exit;
	}

	/* Set maximum tag information; Reserving 1 bit for control information */
	nccl_ofi_comp->max_tag = (uint64_t)((1ULL <<
					    (ofi_tag_bits_for_ring_id - 1)) - 1);

	/* Create fabric */
	ret = fi_fabric(prov->fabric_attr, &(nccl_ofi_comp->fabric), NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric provider. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Create domain */
	ret = fi_domain(nccl_ofi_comp->fabric, prov,
			&(nccl_ofi_comp->domain), NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric access domain. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Create transport level communication endpoint(s) */
	ret = fi_endpoint(nccl_ofi_comp->domain, prov, &(nccl_ofi_comp->ep), NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't allocate endpoint. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	ret = fi_cq_open(nccl_ofi_comp->domain, &cq_attr, &nccl_ofi_comp->cq, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	ret = fi_av_open(nccl_ofi_comp->domain, &av_attr, &nccl_ofi_comp->av, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open AV. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Bind CQ and AV to endpoint */
	ret = fi_ep_bind(nccl_ofi_comp->ep, (fid_t)nccl_ofi_comp->cq, FI_SEND | FI_RECV);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-CQ. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	ret = fi_ep_bind(nccl_ofi_comp->ep, (fid_t)nccl_ofi_comp->av, 0);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-CQ. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Enable endpoint for communication */
	ret = fi_enable(nccl_ofi_comp->ep);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't enable endpoint. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	return ret;
error:
	if (nccl_ofi_comp->domain)
		fi_close((fid_t)nccl_ofi_comp->domain);
	if (nccl_ofi_comp->fabric)
		fi_close((fid_t)nccl_ofi_comp->fabric);
	if (nccl_ofi_comp->ep)
		fi_close((fid_t)nccl_ofi_comp->ep);
	if (nccl_ofi_comp->av)
		fi_close((fid_t)nccl_ofi_comp->av);
	if (nccl_ofi_comp->cq)
		fi_close((fid_t)nccl_ofi_comp->cq);
exit:
	return ret;
}

/*
 * @brief	Allocate and initialize nccl_ofi_component for the given NIC ID
 */
static ncclResult_t create_nccl_ofi_comp_for_dev(int dev, struct fi_info *nic_info_list)
{
	ncclResult_t ret = ncclSuccess;
	struct fi_info *prov = NULL;

	prov = get_nic_info(dev, nic_info_list);
	if (prov == NULL) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Could not extract provider information for given NIC ID %d",
			     dev);
		goto error;
	}

	if (!nccl_ofi_component[dev]) {
		nccl_ofi_component[dev] = (nccl_ofi_t *)calloc(1, sizeof(nccl_ofi_t));
		if (nccl_ofi_component[dev] == NULL) {
			ret = ncclSystemError;
			goto error;
		}
	}

	/* Initialise tag and num_cqes */
	nccl_ofi_component[dev]->tag = 0;
	nccl_ofi_component[dev]->prov_name = prov->fabric_attr->prov_name;

	ret = create_nccl_ofi_component(prov, nccl_ofi_component[dev]);
	if (ret != 0)
		goto error;

	NCCL_OFI_TRACE(NCCL_NET, "OFI component %p for dev #%d is created", nccl_ofi_component[dev], dev);

	return ret;

error:
	if (nccl_ofi_component[dev] != NULL) {
		free(nccl_ofi_component[dev]);
		nccl_ofi_component[dev] = NULL;
	}
	return ret;
}

/*
 * @brief	Release various libfabric resources.
 */
void release_nccl_ofi_component(nccl_ofi_t *nccl_ofi_comp, int dev)
{
	assert(nccl_ofi_comp != NULL);

	if (nccl_ofi_comp->ep)
		fi_close((fid_t)nccl_ofi_comp->ep);
	if (nccl_ofi_comp->av)
		fi_close((fid_t)nccl_ofi_comp->av);
	if (nccl_ofi_comp->cq)
		fi_close((fid_t)nccl_ofi_comp->cq);
	if (nccl_ofi_comp->domain)
		fi_close((fid_t)nccl_ofi_comp->domain);
	if (nccl_ofi_comp->fabric)
		fi_close((fid_t)nccl_ofi_comp->fabric);

	/*
	 * Ideally we would also free up nccl_ofi_comp here but there is no
	 * straightforward way to do that in this case. The caller of
	 * nccl_net_ofi_connect/nccl_net_ofi_listen maintains the
	 * nccl_ofi_component array in its thread local storage and also
	 * allocates memory for individual array entries. The array
	 * entries/individual nccl_ofi_t structures can be used by different
	 * threads which means that the caller of release_nccl_ofi_component
	 * can be different from the caller of nccl_net_ofi_connect or
	 * nccl_net_ofi_listen and that caller has no way of changing the
	 * corresponding array entry in nccl_ofi_component to NULL.
	 * We keep the nccl_ofi_t struct around so that when other threads
	 * find the refcnt to be 0, they know that the libfabric resources need
	 * to be reallocated. In a separate CR we can try storing a pointer to
	 * nccl_ofi_component array in the comm structure. That way other threads
	 * have access to it and can clean up after themselves.
	 */
	NCCL_OFI_TRACE(NCCL_NET, "Libfabric resources for OFI component %p dev #%d is released", nccl_ofi_comp, dev);
}

/*
 * @brief	Get nccl_ofi_component for given device ID.
 * 		Create if it does not exist. Increase refernce counter. Must be
 * 		protected by nccl_ofi_lock.
 */
static nccl_ofi_t *get_nccl_ofi_comp(int dev)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_t *ret_ofi_comp = NULL;

	/*
	 * nccl_ofi_lock is common across all threads even though the data structure
	 * it is protecting, nccl_ofi_component, is local to a particular thread.
	 * While this approach is fine given that we don't call get_nccl_ofi_comp
	 * or put_nccl_ofi_comp in the latency sensitive path, a cleaner approach
	 * would be to break this off into two different locks. A thread local one that
	 * protects the nccl_ofi_component array and individual ones for the nccl_ofi
	 * structures within that can be shared by all threads
	 */
	pthread_mutex_lock(&nccl_ofi_lock);
	if (nccl_ofi_component == NULL) {
		nccl_ofi_component = (nccl_ofi_t **)calloc(ofi_ndevices, sizeof(nccl_ofi_t *));
		if (nccl_ofi_component == NULL) {
			NCCL_OFI_WARN("Unable to allocate nccl_ofi_component");
			goto unlock;
		}
	}
	if (!nccl_ofi_component[dev] || nccl_ofi_component[dev]->refcnt == 0)
		ret = create_nccl_ofi_comp_for_dev(dev, ofi_info_list);

	if (ret == ncclSuccess) {
		++nccl_ofi_component[dev]->refcnt;
		ret_ofi_comp = nccl_ofi_component[dev];
	}
unlock:
	pthread_mutex_unlock(&nccl_ofi_lock);
	return ret_ofi_comp;
}

/*
 * @brief	Release nccl_ofi_component for given device ID.
 *		Decrease refernce counter. Release resources if reference
 *		counter becomes zero. Must be protected by nccl_ofi_lock.
 */
static void put_nccl_ofi_comp(nccl_ofi_t *ofi_comp, int dev)
{
	assert(ofi_comp != NULL);
	pthread_mutex_lock(&nccl_ofi_lock);
	if (--ofi_comp->refcnt == 0)
		release_nccl_ofi_component(ofi_comp, dev);
	pthread_mutex_unlock(&nccl_ofi_lock);
}

#define __compiler_barrier() do { asm volatile ("" : : : "memory"); } while(0)

/*
 * @brief	Update nccl_ofi_req on completion
 *		Fill up request context to deliver to user along with state update.
 *		User polls state field to check completion.
 *
 */
static inline void update_nccl_ofi_req(nccl_ofi_req_t *req, nccl_ofi_req_state_t state, size_t size)
{
	req->size = size;
	/* As nccl_net_ofi_test() can be called on other thread, state should
	 * be updated last and there should be a barrier before state update */
	__sync_synchronize();
	req->state = state;
}

/*
 * @brief	Processes completion entries from CQ
 *
 * @return	0, on success
 *		error, on others
 */
static inline ncclResult_t process_completions(
				struct fi_cq_tagged_entry *cq_entry,
				uint64_t num_cqes, uint64_t control_bit_mask)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_req_t *req = NULL;
	uint64_t comp_idx = 0, comp_flags = 0;

	for (comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		void *op_ctx = cq_entry[comp_idx].op_context;

		if (OFI_UNLIKELY(op_ctx == NULL)) {
			NCCL_OFI_WARN("Invalid request context provided");
			ret = ncclSystemError;
			goto exit;
		}

		comp_flags = cq_entry[comp_idx].flags;

		req = container_of(op_ctx, nccl_ofi_req_t, ctx);
		update_nccl_ofi_req(req, NCCL_OFI_REQ_COMPLETED, cq_entry[comp_idx].len);

		NCCL_OFI_TRACE_COMPLETIONS(req, &req->ctx);

		/* Determine if this is control message */
		if (OFI_UNLIKELY(cq_entry[comp_idx].tag & control_bit_mask)) {
			if (comp_flags & FI_RECV) {
				/* Mark listenComm to accepted state */
				req->lComm->accepted = true;
			}
		}
	}

exit:
	return ret;
}

/*
 * @brief	Process completion entries for the given NCCL OFI component.
 *		This also updates several request fileds like size, status, etc
 *
 * @return	0, on success
 *		error, on others
 */
static ncclResult_t ofi_process_cq(nccl_ofi_t *nccl_ofi_comp)
{
	ssize_t rc = 0;
	ncclResult_t ret = ncclSuccess;
	struct fi_cq_err_entry err_buffer = { 0 };
	struct fi_cq_tagged_entry cqe_tagged_buffers[cq_read_count];
	nccl_ofi_req_t *req = NULL;
	struct fid_cq *cq = nccl_ofi_comp->cq;
	uint64_t control_bit_mask = nccl_ofi_comp->max_tag + 1;

	while (true) {
		/* Receive completions for the given endpoint */
		rc = fi_cq_read(cq, cqe_tagged_buffers, cq_read_count);
		if (rc > 0) {
			ret = process_completions(
					cqe_tagged_buffers, rc,
					control_bit_mask);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
		}
		else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
			rc = fi_cq_readerr(cq, &err_buffer, 0);
			if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
				/*
				 * Error not available yet.
				 * fi_cq_read will keep returning -FI_EAVAIL so just bail out and try again later.
				 */
				break;
			} else if (OFI_UNLIKELY(rc < 0)) {
				NCCL_OFI_WARN("Unable to read from fi_cq_readerr. RC: %zd. Error: %s",
					     rc,
					     fi_strerror(-rc));
				ret = ncclSystemError;
				goto exit;
			}

			req = container_of(err_buffer.op_context,
					   nccl_ofi_req_t, ctx);
			NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %s. Completed length: %ld, Request: %s",
					req,
					err_buffer.err,
				    fi_cq_strerror(cq,
						err_buffer.prov_errno,
						err_buffer.err_data, NULL, 0),
					(long)err_buffer.len,
					nccl_ofi_request_str(req));
			update_nccl_ofi_req(req, NCCL_OFI_REQ_ERROR, err_buffer.len);
		}
		else if (rc == -FI_EAGAIN) {
			/* No completions to process */
			break;
		}
		else {
			NCCL_OFI_WARN("Unable to retrieve completion queue entries. RC: %zd, ERROR: %s",
				     rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto exit;
		}
	}

exit:
	return ret;
}

static inline ncclResult_t nccl_ofi_progress(nccl_ofi_t *nccl_ofi_comp)
{
	if (OFI_UNLIKELY(nccl_ofi_comp == NULL)) {
		NCCL_OFI_WARN("NCCL OFI component is not initialised");
		return ncclSystemError;
	}
	/* Read completion queue entries */
	return ofi_process_cq(nccl_ofi_comp);
}

ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction)
{
	ncclResult_t ret = ncclSuccess;

	ofi_log_function = logFunction;

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using " PACKAGE_STRING);

#if HAVE_CUDA
	if (ofi_nccl_cuda_flush_enable()) {
#if CUDART_VERSION < 11030
		NCCL_OFI_WARN("CUDA flush requested, but CUDART_VERSION %ld < 11030", CUDART_VERSION);
		cuda_flush = false;
#else
		NCCL_OFI_WARN("CUDA flush enabled");
		cuda_flush = true;
#endif
	}
#endif

	nic_dup_conns = ofi_nccl_nic_dup_conns();

	if (platform_init) {
		ret = platform_init();
		if (ret != ncclSuccess)
			goto exit;
	}

	/* Get list of NICs fi_info structures for a single provider */
	ret = get_ofi_provider(provider_filter, &ofi_info_list);
	if (ret != 0 || ofi_info_list == NULL) {
		ret = ncclSystemError;
		goto exit;
	}

	/* Allow for multiple virtual nics per nic to increase
	 * throughput for NICs that do not handle single QP situations
	 * well. */
	if (nic_dup_conns > 1 && !support_gdr) {
		ofi_ndevices *= nic_dup_conns;

		// Make the list cyclic to emulate having multiple devices
		ofi_info_list->next = ofi_info_list;
		NCCL_OFI_INFO(NCCL_INIT, "DUP_CONNS of %d changing device count to %d",
			      nic_dup_conns, ofi_ndevices);
	} else if (nic_dup_conns > 0) {
		NCCL_OFI_WARN("NCCL_OFI_NIC_DUP_CONNS set on platform that supports GPUDirect RDMA.  This configuration is not supported.");
		ret = ncclSystemError;
		goto exit;
	}

	/* If TCP provider is selected, filter out unnecessary interfaces and address formats */
	if (strncmp("tcp", ofi_info_list->fabric_attr->prov_name, strlen("tcp")) == 0) {
		filter_tcp_info_list();
		if (ofi_info_list == NULL) {
			NCCL_OFI_WARN("No viable endpoint found for TCP provider. Try and relax the filters using OFI_NCCL_USE_IPV6_TCP or OFI_NCCL_EXCLUDE_TCP_IF environment variables");
			ret = ncclSystemError;
			goto exit;
		}
	}

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Selected Provider is %s (found %d nics)",
		      ofi_info_list->fabric_attr->prov_name, ofi_ndevices);

	/* Check if provider requires local memory registration */
	if (ofi_info_list->domain_attr->mr_mode & FI_MR_LOCAL) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s requires registration of local memory buffers",
			       ofi_info_list->fabric_attr->prov_name);
		local_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not require registration of local memory buffers",
			       ofi_info_list->fabric_attr->prov_name);
	}

	/* Check if provider uses remote virtual addressing */
	if (ofi_info_list->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s uses remote virtual addressing",
			       ofi_info_list->fabric_attr->prov_name);
		virt_addr_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not use remote virtual addressing",
			       ofi_info_list->fabric_attr->prov_name);
	}

	/* Check if provider selects memory registration keys */
	if (!(ofi_info_list->caps & FI_RMA)) {
		/* When FI_RMA is not requested, Libfabric considers
		   memory registrations to be local only, and
		   therefore the requested_key field is ignored and
		   (unfortunately) a random key may be returned from
		   fi_mr_key().  This totally screws up the code to
		   provide a unique MR key, which is, according to
		   Libfabric, unnecessary in this mode anyway, so fall
		   back to the provider-specified key code, which
		   should behave properly in either case. */
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s only configured for local registration.",
			       ofi_info_list->fabric_attr->prov_name);
		prov_key_mr = true;
	} else if (ofi_info_list->domain_attr->mr_mode & FI_MR_PROV_KEY) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s selects memory registration keys",
			       ofi_info_list->fabric_attr->prov_name);
		prov_key_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not select memory registration keys",
			       ofi_info_list->fabric_attr->prov_name);

		if (ofi_info_list->domain_attr->mr_key_size < ofi_nccl_mr_key_size()) {
			NCCL_OFI_WARN("Provider %s supports MR key size of %zu, but %zu was requested",
				      ofi_info_list->fabric_attr->prov_name,
				      ofi_info_list->domain_attr->mr_key_size,
				      ofi_nccl_mr_key_size());

			ret = ncclSystemError;
			goto exit;
		}

		/* The provider may return support for a larger key size. Use
		 * the size requested by the user to allow them to limit the
		 * size of the mr_keys table. */
		num_mr_keys = (size_t) 1 << (ofi_nccl_mr_key_size() * 8);

		mr_keys = malloc(sizeof(bool) * num_mr_keys * ofi_ndevices);
		if (NULL == mr_keys) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Unable to allocate MR keys table");
			ret = ncclSystemError;
			goto exit;
		}
		for (size_t i = 0; i < num_mr_keys * ofi_ndevices; i++)
			mr_keys[i] = true;
	}

	/* Check if provider uses endpoint memory registration */
	if (ofi_info_list->domain_attr->mr_mode & FI_MR_ENDPOINT) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s requires endpoint memory registration",
			       ofi_info_list->fabric_attr->prov_name);
		endpoint_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not require endpoint memory registration",
			       ofi_info_list->fabric_attr->prov_name);
	}

	/* Store the cq_read_count parameter value in a global
	   variable to avoid the lookup overhead during execution. */
	cq_read_count = ofi_nccl_cq_read_count();

exit:
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN(PACKAGE_NAME " initialization failed");
	}
	return ret;
}

ncclResult_t nccl_net_ofi_devices(int *ndev)
{
	*ndev = ofi_ndevices;
	return ncclSuccess;
}

static ncclResult_t get_device_pci_path(int dev, struct fid_nic *nic_info, char** path)
{
	ncclResult_t ret = ncclSuccess;
	struct fi_pci_attr *pci = NULL;
	char *device_path = NULL;
	int ret_int;

	if (nic_info->bus_attr->bus_type != FI_BUS_PCI) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Invalid type of PCI bus returned %d",
			      nic_info->bus_attr->bus_type);
		ret = ncclSystemError;
		goto exit;
	}

	pci = &nic_info->bus_attr->attr.pci;
	ret_int = asprintf(&device_path,
			   "/sys/class/pci_bus/%04x:%02x/../../%04x:%02x:%02x.%01x",
			   pci->domain_id, pci->bus_id,
			   pci->domain_id, pci->bus_id, pci->device_id, pci->function_id);
	if (ret_int < 0) {
		NCCL_OFI_WARN("pciPath: Allocation failure");
		ret = ncclSystemError;
		goto exit;
	}

	*path = realpath(device_path, NULL);
	if (*path == NULL) {
		NCCL_OFI_WARN("pciPath: Could not find real path of %s",
			      device_path);
		ret = ncclSystemError;
		goto exit;
	}

exit:
	if (device_path)
		free(device_path);

	return ret;
}

static ncclResult_t set_nic_props_default(int dev, struct fi_info *nic_prov,
					  ncclNetProperties_t *props)
{
	ncclResult_t ret = ncclSuccess;

	props->name = strdup(nic_prov->domain_attr->name);

	/*
	 * Currently, libfabric providers provide multiple `fi_info`
	 * objects for devices with multiple ports. So, safely assume port number
	 * to be always 1.
	 */
	props->port = 1;
	props->maxComms = nic_prov->domain_attr->ep_cnt;
	props->guid = dev;

	if (IS_EFA_PROVIDER(nic_prov->fabric_attr->prov_name)) {
		/*
		 * Sets intranode latency for EFA networks.
		 *
		 * This value is chosen by measuring all reduce latency for
		 * different NCCL algorithms and using that to calculate intra node
		 * latency based on NCCL's tuning algorithm.
		 *
		 * A few different values around this value were tried to see which
		 * chose the correct algorithm (tree or ring) most times across
		 * different message and cluster sizes.
		 */
		props->latency = 150;
	}

	/*
	 * Maximum number of grouped receives. Currently, we set it to 1 to
	 * maintain single send/recv semantics (similar to NCCL versions < v2.12).
	 *
	 * Grouped receives are useful for alltoall collectives where one
	 * receiver is expected to receive from multiple remote GPUs using
	 * PXN(PCIe X NVLINK) feature. Other collectives like allreduce aren't
	 * impacted with this feature as NCCL doesn't aggregate receives from
	 * same source.
	 */
	props->maxRecvs = NCCL_OFI_MAX_RECVS;

	props->ptrSupport = NCCL_PTR_HOST;
	if (support_gdr) {
		/* Supports message transfer from both CUDA and HOST buffers */
#if HAVE_CUDA
		props->ptrSupport |= NCCL_PTR_CUDA;
#elif HAVE_NEURON
		props->ptrSupport |= NCCL_PTR_NEURON;
#endif
	}

	/* Should be successful for ptrSupport invocation */
	return ret;
}

ncclResult_t nccl_net_ofi_getProperties(int dev, ncclNetProperties_t *props)
{
	ncclResult_t ret = ncclSuccess;
	ncclNetProperties_t dev_props = {0};
	struct fi_info *nic_prov = NULL;
	struct fid_nic *nic_info = NULL;

	if (dev < 0 || dev >= ofi_ndevices) {
		NCCL_OFI_WARN("Incorrect dev %d provided", dev);
		ret = ncclInternalError;
		goto error;
	}

	nic_prov = get_nic_info(dev, ofi_info_list);
	if (nic_prov == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Unable to find provider for dev %d", dev);
		ret = ncclSystemError;
		goto error;
	}

	ret = set_nic_props_default(dev, nic_prov, &dev_props);
	if (ret != ncclSuccess)
		goto error;

	/* Change default values as set by NIC attributes */
	nic_info = (struct fid_nic *)nic_prov->nic;
	if (nic_info == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "No NIC info for dev %d. Supplying default values for NIC properties.",
			      dev);
		goto exit;
	}

	/* name is NULL if device is a part of multirail config */
	/* overriding default name only if value is available from provider */
	if (nic_info->device_attr->name) {
		dev_props.name = strdup(nic_info->device_attr->name);
	}

	/* Speed reported in Mbps */
	dev_props.speed = nic_info->link_attr->speed / (1e6);

	ret = get_device_pci_path(dev, nic_info, &(dev_props.pciPath));
	if (ret != ncclSuccess)
		props->pciPath = NULL;

	if (nic_dup_conns > 1) {
#if HAVE_CUDA
		int num_gpus_visible, active_cuda_device, gpus_per_conn;
		size_t c;

		if (cudaGetDeviceCount(&num_gpus_visible) != cudaSuccess) {
			NCCL_OFI_WARN("Error getting CUDA device count");
			ret = ncclUnhandledCudaError;
			goto error;
		}

		if (cudaGetDevice(&active_cuda_device) != cudaSuccess) {
			NCCL_OFI_WARN("Error getting current CUDA device");
			ret = ncclUnhandledCudaError;
			goto error;
		}

		gpus_per_conn = num_gpus_visible / ofi_ndevices;
		if (gpus_per_conn == 0) gpus_per_conn = 1;

		/* The goal is to have gpus_per_conn gpus in the local
		 * system think that any given virtual nic is the one
		 * that they should use, and that it is different than
		 * the other NICs in the system.  We do this by only
		 * leaving a valid device id in pciPath when
		 * active_cuda_device / gpus_per_comm is equal to the
		 * NIC dev index we're currently querying.  For the
		 * others, we provide a PCIPath that points at the PCI
		 * Bus itself, which NCCL will interpret to be on a
		 * different complex than the bus (assuming the NIC
		 * BUS and GPU BUS are the same).
		 *
		 * There are a bunch of assumptions in this logic,
		 * such that the physical NICs in the system don't
		 * have PCI affinity with the GPUs.  Given that we've
		 * already established that GPUDirect doesn't work,
		 * this is probably ok; any affinity is lost by
		 * bouncing through host buffers anyway.
		 */
		if (active_cuda_device / gpus_per_conn  != dev) {
			for (c=strlen(dev_props.pciPath); c && dev_props.pciPath[c] != '/'; c--) {
				dev_props.pciPath[c] = '\0';
			}
			dev_props.pciPath[c] = '\0';
		}
		NCCL_OFI_TRACE(NCCL_INIT, "Returning synthetic PCI path for device %d of  %s",
			       dev, dev_props.pciPath);

		snprintf(dev_props.name, FI_NAME_MAX + 2, "%s-%x", nic_info->device_attr->name, dev);
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Adjusted dev %d device name to %s",
			       dev, dev_props.name);
#else
		NCCL_OFI_WARN("NIC_DUP_CONNS enabled on platform that does not support NIC_DUP_CONNS.  This should not happen.");
		ret = ncclSystemError;
		goto error;
#endif
	}

	goto exit;

error:
	props = NULL;
exit:
	*props = dev_props;
	return ret;
}

/*
 * @brief	Query local address for a libfabric endpoint
 *
 * @param	Network device
 *
 * @return	Local EP address, on success
 * 		NULL, others
 */
static inline char *get_local_address(int dev, nccl_ofi_t *nccl_ofi_comp)
{
	int ret = 0;
	size_t namelen = MAX_EP_ADDR;
	char *local_ep_addr = (char *)calloc(namelen, sizeof(char));

	ret = fi_getname(&(nccl_ofi_comp->ep->fid),
			(void *)local_ep_addr,
			&namelen);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%d) is larger than supplied buffer length (%d)",
			      namelen, MAX_EP_ADDR);
		free(local_ep_addr);
		return NULL;
	} else if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		free(local_ep_addr);
		return NULL;
	}

	return local_ep_addr;
}


ncclResult_t nccl_net_ofi_listen(int dev, void *handle, void **listenComm)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_handle_t *ofi_handle = (nccl_ofi_handle_t *)handle;
	char *local_ep_name = NULL;
	fi_addr_t local_ep_addr;
	listenComm_t *lComm = NULL;
	uint64_t tag;
	int num_addrs;
	nccl_ofi_t *nccl_ofi_comp = NULL;

	if (OFI_UNLIKELY(dev < 0 || dev >= ofi_ndevices)) {
		NCCL_OFI_WARN("Incorrect device ID %d provided. Correct values are from 0 to %d",
			      dev, ofi_ndevices - 1);
		ret = ncclInternalError;
		goto exit;
	}

	/* Zero-out the handle */
	memset(ofi_handle, 0, sizeof(nccl_ofi_handle_t));

	/*
	 * Create libfabric components for the given NIC, if not
	 * already created, else increase tag ID.
	 */
	nccl_ofi_comp = get_nccl_ofi_comp(dev);
	if (!nccl_ofi_comp)
		goto exit;

	if (nccl_ofi_comp->tag + 1 >=
	    nccl_ofi_comp->max_tag) {
		NCCL_OFI_WARN("Cannot open more connection for device ID %d."
			      " Maximum is %ld",
			      dev, nccl_ofi_comp->max_tag);
		ret = ncclSystemError;
		goto error;
	}
	tag = ++nccl_ofi_comp->tag;

	/* Build handle */
	local_ep_name = get_local_address(dev, nccl_ofi_comp);

	memcpy(ofi_handle->ep_name, local_ep_name, MAX_EP_ADDR);
	ofi_handle->tag = tag;

	/* Insert local EP address to AV. This will be used to issue local read operations */
	num_addrs = fi_av_insert(nccl_ofi_comp->av, (void *)local_ep_name, 1,
				 &local_ep_addr, 0, NULL);
	if (OFI_UNLIKELY(num_addrs != 1)) {	/* Only 1 address should be inserted into the AV */
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      dev, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	} else {
		ret = ncclSuccess;
	}

	/* Build listenComm */
	lComm = (listenComm_t *)calloc(1, sizeof(listenComm_t));
	if (OFI_UNLIKELY(lComm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate listenComm for dev %d", dev);
		ret = ncclSystemError;
		goto error;
	}
	lComm->tag = tag;
	lComm->local_ep = nccl_ofi_comp->ep;
	lComm->accepted = false;
	lComm->dev = dev;
	lComm->local_ep_addr = local_ep_addr;
	lComm->baseComm.ofi_comp = nccl_ofi_comp;

	*listenComm = lComm;

	goto exit;

error:
	if (lComm)
		free(lComm);
	if (nccl_ofi_comp)
		put_nccl_ofi_comp(nccl_ofi_comp, dev);
exit:
	return ret;
}

/*
 * @brief	Creates send communication for a peer
 *
 * @param	Network device ID
 * 		Connection Handle transferred OOB by NCCL
 *
 * @return	Initialized Send Communicator object, on success
 * 		NULL, others
 * @return	0, success
 * 		error, others
 *
 */
static inline int create_sendComm(int dev, nccl_ofi_handle_t *ofi_handle, nccl_ofi_t *nccl_ofi_comp, sendComm_t **sendComm)
{
	char remote_ep_addr[MAX_EP_ADDR] = {0};
	uint64_t tag = 0ULL;
	uint64_t max_tag = 0;
	size_t req_size = sizeof(nccl_ofi_req_t);
	fi_addr_t remote_addr;
	sendComm_t *sComm = NULL;
	*sendComm = NULL;

	/*
	 * Create libfabric components for the given NIC, if not
	 * already created.
	 */
	max_tag = nccl_ofi_comp->max_tag;

	/* Get tag and remote name from handle */
	memcpy(&remote_ep_addr, ofi_handle->ep_name, MAX_EP_ADDR);
	memcpy(&tag, &ofi_handle->tag, sizeof(tag));
	if (tag < 1 || tag > max_tag) {
		NCCL_OFI_WARN("Received an invalid tag %lu for device %d", tag,
				dev);
		return ncclSystemError;
	}

	ncclResult_t ret = ncclSuccess;

	/* Insert remote address into AV */
	ret = fi_av_insert(nccl_ofi_comp->av,
			   (void *)remote_ep_addr, 1,
			   &remote_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
				dev, ret);
		return ncclSystemError;
	}

	/* Allocate and initialize sendComm */
	sComm = (sendComm_t *)calloc(1, sizeof(sendComm_t));
	if (OFI_UNLIKELY(sComm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate sendComm for dev %d", dev);
		return ncclSystemError;
	}

	sComm->tag = tag;
	sComm->local_ep = nccl_ofi_comp->ep;
	sComm->remote_ep = remote_addr;
	sComm->dev = dev;
	sComm->baseComm.ofi_comp = nccl_ofi_comp;

	sComm->connection_info = calloc(1, sizeof(nccl_ofi_connection_info_t));
	if (!sComm->connection_info) {
		return ncclSystemError;
	}

	sComm->connection_info->ep_namelen = sizeof(sComm->connection_info->ep_name);

	ret = fi_getname(&(sComm->baseComm.ofi_comp->ep->fid),
			 (void *)sComm->connection_info->ep_name,
			 &sComm->connection_info->ep_namelen);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%d) is larger than supplied buffer length (%d)",
			      sComm->connection_info->ep_namelen, MAX_EP_ADDR);
		return ncclSystemError;
	} else if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ncclSystemError;
	}

	sComm->connection_info->connect_to_self =
		(0 == memcmp(sComm->connection_info->ep_name, remote_ep_addr, sComm->connection_info->ep_namelen)) ? 1 : 0;

	/* Pre-allocated buffers for data path */
	ret = allocate_ofi_fl(&sComm->nccl_ofi_reqs_fl, max_requests, req_size);
	if (OFI_UNLIKELY(ret != ncclSuccess)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
				dev);
		free(sComm);
		return ret;
	}

	*sendComm = sComm;
	return ret;
}

/*
 * @brief	Prepare a send request for a given sendComm
 *
 * @param	Valid send communicator object
 *
 * @return	NCCL OFI request, on success
 * 		NULL, others
 */
static inline nccl_ofi_req_t *prepare_send_req(sendComm_t *sComm)
{
	nccl_ofi_req_t *req = NULL;

	req = allocate_nccl_ofi_request(sComm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      sComm->dev);
		return NULL;
	}

	req->sComm = sComm;
	req->dev = sComm->dev;
	req->direction = NCCL_OFI_SEND;

	return req;
}

/*
 * @brief	Send connect request to send communicator's peer
 *
 * @param	Valid send communicator object
 * 		NCCL OFI request
 *
 * @return	0, on successfully sending message
 * 		-1, on failure to get local EP address
 * 		-FI_EAGAIN, on lack of provider resources to send message
 * 		others, on error
 */
static ssize_t send_connect_message(sendComm_t *sComm, nccl_ofi_req_t *req)
{
	ssize_t rc = 0;
	int ret = ncclSuccess;
	uint64_t max_tag = sComm->baseComm.ofi_comp->max_tag;

	/* If connecting to self, pass along the send req so that the
	   accept side can clean up the request */
	sComm->connection_info->req = (sComm->connection_info->connect_to_self == 1) ? req : NULL;

	/*
	 * TODO: replace it with API of FI_INJECT type when most of
	 * providers can support it, so that need for completion check
	 * can be lifted.
	 */
	rc = fi_tsend(sComm->local_ep, (void *)sComm->connection_info,
		      sizeof(*sComm->connection_info), NULL, sComm->remote_ep,
		      sComm->tag | (max_tag + 1), &req->ctx);
	if (rc == -FI_EAGAIN) {
		/*
		 * Process completions so that you have enough
		 * resources for sending connect message
		 */
		ret = nccl_ofi_progress(sComm->baseComm.ofi_comp);
		if (ret != ncclSuccess)
			return ncclSystemError;
	} else if (rc != 0) {
		NCCL_OFI_WARN("Unable to send connect message for dev %d. RC: %zd, ERROR: %s",
			       sComm->dev, rc, fi_strerror(-rc));
	}

	return rc;
}

/*
 * @brief	Non-blocking connect which returns sendComm as NULL
 *		with an expectation that it will be called again until sendComm != NULL
 *
 * @param	Network Device ID
 * 		Connection Handle (transferred OOB by NCCL)
 *
 * @return	sendComm = NULL, if connection hasn't been established
 * 		sendComm != NULL, once connection is established
 * @return	0, on success
 * 		error, on others
 */
ncclResult_t nccl_net_ofi_connect(int dev, void *handle, void **sendComm)
{
	ncclResult_t ret = ncclSuccess;
	ssize_t rc = 0;

	*sendComm = NULL;

	if (OFI_UNLIKELY(dev < 0 || dev >= ofi_ndevices)) {
		NCCL_OFI_WARN("Incorrect device ID %d provided. Correct values are from 0 to %d",
			      dev, ofi_ndevices - 1);
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(handle == NULL)) {
		NCCL_OFI_WARN("Provided handle is NULL");
		return ncclSystemError;
	}

	nccl_ofi_handle_t *ofi_handle = (nccl_ofi_handle_t *)handle;

	/* Extract connection state of the communicator */
	save_comm_state_t *comm_state = &(ofi_handle->state);
	nccl_ofi_req_t *req = comm_state->req;
	sendComm_t *sComm = comm_state->comm;

	/*
	 * Take appropriate actions based on connection stage of communicator.
	 *
	 * Once we have completed the actions for a particular stage, we proceed
	 * to the next one until failure. This is to ensure we make maximum
	 * progress in a single function invocation.
	 */
	nccl_ofi_comm_stage_t stage = comm_state->stage;
	switch (stage) {
		case COMM_CREATE_START:
			/*
			 * When we are building the sComm for the first time,
			 * it should *NOT* come initialized from handle.
			 */
			assert(sComm == NULL);

			nccl_ofi_t *nccl_ofi_comp = get_nccl_ofi_comp(dev);
			if (!nccl_ofi_comp) {
				return ncclSuccess;
			}

			/* Build sendComm */
			ret = create_sendComm(dev, ofi_handle, nccl_ofi_comp, &sComm);
			if (OFI_UNLIKELY(ret != ncclSuccess)) {
				put_nccl_ofi_comp(nccl_ofi_comp, dev);
				return ret;
			}

			/* Prepare connect request to be sent to peer */
			req = prepare_send_req(sComm);
			if (OFI_UNLIKELY(req == NULL)) {
				free(sComm);
				put_nccl_ofi_comp(nccl_ofi_comp, dev);
				return ncclSystemError;
			}

			comm_state->stage = COMM_SEND_CONN;

		case COMM_SEND_CONN:
			/* Send "connect" message to remote EP */
			rc = send_connect_message(sComm, req);
			if (rc == -FI_EAGAIN) {
				/* Save connection state */
				comm_state->comm = sComm;
				comm_state->req = req;
				return ncclSuccess;
			}
			else if (rc != 0) {
				put_nccl_ofi_comp(sComm->baseComm.ofi_comp, dev);
				free(sComm);
				free_nccl_ofi_req(req, false);
				return ncclSystemError;
			}

			comm_state->stage = COMM_REQ_PENDING_COMP;

		case COMM_REQ_PENDING_COMP:
			if (sComm->connection_info->connect_to_self == 1) {
				NCCL_OFI_TRACE(NCCL_NET, "Connect to self; short circuit cleanup");
				/* short cut to avoid rendezvous
				   deadlock in GDR detection */
				comm_state->stage = COMM_CONNECTED;
				break;
			}

			/* Progress our engine to get completions */
			ret = nccl_ofi_progress(sComm->baseComm.ofi_comp);
			if (OFI_UNLIKELY(ret != ncclSuccess)) {
				put_nccl_ofi_comp(sComm->baseComm.ofi_comp, dev);
				free(sComm);
				free_nccl_ofi_req(req, false);
				return ncclSystemError;
			}

			/* Check if the connect message is sent. */
			if (req->state != NCCL_OFI_REQ_COMPLETED) {
				/* Save connection state */
				comm_state->comm = sComm;
				comm_state->req = req;
				return ncclSuccess;
			}

			comm_state->stage = COMM_CONNECTED;

			break;

		case COMM_RECV_CONN:
		case COMM_CONNECTED:
		default:
			NCCL_OFI_WARN("Invalid state of send communicator object: %d", stage);
			return ncclSystemError;
	};

	*sendComm = sComm;
	free_nccl_ofi_req(req, false);

	return ret;
}

/*
 * @brief	Allocate a request to receive peer connection message
 *
 * @param	Valid listen communicator object
 *
 * @return	NCCL OFI request, on success
 * 		NULL, on error
 */
static nccl_ofi_req_t *prepare_recv_conn(listenComm_t *lComm)
{
	nccl_ofi_req_t *req = NULL;

	/* Allocate a NCCL OFI request */
	req = (nccl_ofi_req_t *)calloc(1, sizeof(nccl_ofi_req_t));
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to allocate nccl_ofi_req_t");
		return NULL;
	}

	req->state = NCCL_OFI_REQ_CREATED;
	req->lComm = lComm;
	req->dev = lComm->dev;

	return req;
}

/*
 * @brief	Post a request to receive peer connection message
 *
 * @param	listen communicator object, contains the local EP and device information
 * 		buffer, to receive connection message
 * 		NCCL OFI receive request
 *
 * @return	0, on successful posting of receive request
 * 		-FI_EAGAIN, on lack of provider resources to post receive request
 * 		error, others
 */
static ssize_t post_recv_conn(listenComm_t *lComm, nccl_ofi_t *nccl_ofi_comp, void *buffer,
			      size_t size, nccl_ofi_req_t *req)
{
	ssize_t rc = 0;
	int ret = ncclSuccess;
	int dev = lComm->dev;
	uint64_t max_tag;

	if (nccl_ofi_comp == NULL) {
		NCCL_OFI_WARN("NCCL OFI component for dev %d is uninitialised",
			      dev);
		return ncclSystemError;
	}

	max_tag = nccl_ofi_comp->max_tag;

	/* Post a buffer for receiving connection requests */
	rc = fi_trecv(lComm->local_ep, buffer, size,
		      NULL, FI_ADDR_UNSPEC, lComm->tag | (max_tag + 1),
		      0, &req->ctx);
	if (rc == -FI_EAGAIN) {
		/*
		 * Process completions so that you have enough
		 * resources for posting receive buffer
		 */
		ret = nccl_ofi_progress(nccl_ofi_comp);
		if (OFI_UNLIKELY(ret != 0))
			return ncclSystemError;
	}
	else if (rc != 0)
		NCCL_OFI_WARN("Unable to post a buffer for receving connections for dev %d. RC: %zd, ERROR: %s",
			      dev, rc, fi_strerror(-rc));

	return rc;
}

/*
 * @brief	Allocated and registers buffer to flush RDMA operations. On
 * 		Success, receive communicator holds reference to flush buffer
 * 		and associated memory handle.
 *
 * @param	Valid receive communicator object
 *
 * @return	0, on success
 * 		error, on others
 */
static int alloc_and_reg_flush_buff(recvComm_t *rComm)
{
	int ret = ncclSuccess;
	const long page_size = sysconf(_SC_PAGESIZE);
	struct fid_mr *mr_handle = NULL;

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Registering buffer for flush operations");

	rComm->flush_buff.host_buffer = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
					     MAP_PRIVATE | MAP_ANON, -1, 0);
	if (OFI_UNLIKELY(rComm->flush_buff.host_buffer == MAP_FAILED)) {
		NCCL_OFI_WARN("Unable to allocate flush buffer (%d %s)",
			      errno, strerror(errno));
		return ncclSystemError;
	}

	/* Register flush dummy buffer for provider access */
	ret = register_mr_buffers(rComm, rComm->flush_buff.host_buffer,
				  page_size, NCCL_PTR_HOST,
				  &mr_handle);
	if (OFI_UNLIKELY(ret != ncclSuccess)) {
		NCCL_OFI_WARN("Could not register dummy buffer for flush, dev: %d",
				rComm->dev);
		if (munmap(rComm->flush_buff.host_buffer, page_size)) {
			NCCL_OFI_WARN("Unable to unmap flush buffer (%d %s)",
				      errno, strerror(errno));
		}
		rComm->flush_buff.host_buffer = MAP_FAILED;
	}

	rComm->flush_buff.mr_handle = mr_handle;

	return ret;
}

/*
 * @brief	Allocate and setup receive communicator object for a peer. This
 * 		prepares plugin to receive messages from the given peer.
 *
 * @param	Valid listen communicator object
 * 		Peer address
 *
 * @return	Receive communicator object, on success
 * 		NULL, on error
 */
static recvComm_t *prepare_recv_comm(listenComm_t *lComm, char *remote_ep_addr)
{
	int ret = ncclSuccess;
	nccl_ofi_t *nccl_ofi_comp = lComm->baseComm.ofi_comp;
	fi_addr_t remote_ep;
	recvComm_t *rComm = NULL;
	size_t req_size = sizeof(nccl_ofi_req_t);

	/* Insert remote EP address to AV */
	ret = fi_av_insert(nccl_ofi_comp->av, (void *)remote_ep_addr, 1,
			   &remote_ep, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      lComm->dev, fi_strerror(-ret));
		return NULL;
	}

	/* Build recvComm */
	rComm = (recvComm_t *)calloc(1, sizeof(recvComm_t));
	if (rComm == NULL) {
		NCCL_OFI_WARN("Unable to allocate receive Comm object for device %d",
			      lComm->dev);
		return NULL;
	}

	rComm->tag = lComm->tag;
	rComm->local_ep = lComm->local_ep;
	rComm->local_ep_addr = lComm->local_ep_addr;
	rComm->remote_ep = remote_ep;
	rComm->dev = lComm->dev;
	rComm->baseComm.ofi_comp = lComm->baseComm.ofi_comp;

	/* Pre-allocated buffers for data path */
	ret = allocate_ofi_fl(&rComm->nccl_ofi_reqs_fl, max_requests, req_size);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
			     lComm->dev);
		free(rComm);
		return NULL;
	}

	/*
	 * Setup flush resources if using GPUDirect RDMA unless user disables
	 * flush operations
	 */
	if (!ofi_nccl_gdr_flush_disable() && support_gdr && !cuda_flush) {
		rComm->flush_buff.size = NCCL_OFI_FLUSH_SIZE;
		ret = alloc_and_reg_flush_buff(rComm);
		if (OFI_UNLIKELY(ret != ncclSuccess)) {
			free(rComm);
			return NULL;
		}
	}

	return rComm;
}

/*
 * @brief	Non-blocking accept which returns recvComm as NULL
 * 		with an expectation that it will be called again until
 * 		recvComm != NULL
 *
 * @param	Listen Communicator object
 *
 * @return	recvComm = NULL, if connection hasn't been established
 * 		recvComm != NULL, once connection is established
 * @return	0, on success
 * 		error, on others
 */
ncclResult_t nccl_net_ofi_accept(void *listenComm, void **recvComm)
{
	ncclResult_t ret = ncclSuccess;
	ssize_t rc = 0;

	listenComm_t *lComm = (listenComm_t *)listenComm;
	if (lComm == NULL) {
		NCCL_OFI_WARN("Invalid listen communicator provided");
		return ncclInternalError;
	}
	int dev = lComm->dev;

	if (lComm->state.stage != COMM_REQ_PENDING_COMP && lComm->accepted) {
		NCCL_OFI_WARN("listenComm %p object already has an active connection (%d).", listenComm, lComm->accepted);
		return ncclSystemError;
	}

	*recvComm = NULL;

	/* Extract communicator state from listen communicator object */
	save_comm_state_t *comm_state = &lComm->state;
	recvComm_t *rComm = comm_state->comm;
	nccl_ofi_req_t *req = comm_state->req;

	/* Extract peer address from listen communicator's buffer */
	nccl_ofi_connection_info_t *conn_info = lComm->buffer;

	nccl_ofi_t *nccl_ofi_comp = lComm->baseComm.ofi_comp;
	if (nccl_ofi_comp == NULL) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("NCCL OFI component for dev %d is uninitialised",
			     dev);
		return ret;
	}


	/*
	 * Take appropriate actions based on connection stage of communicator.
	 *
	 * Once we have completed the actions for a particular stage, we proceed
	 * to the next one until failure. This is to ensure we make maximum
	 * progress in a single function invocation.
	 */
	nccl_ofi_comm_stage_t stage = comm_state->stage;
	switch (stage) {
		case COMM_CREATE_START:

			/*
			 * The libfabric resources maintained by the ofi_comp
			 * structure is passed from lComm to rComm so they can
			 * then be used by nccl_net_ofi_irecv. We want to make
			 * sure those resources are not freed up when we call
			 * nccl_net_ofi_closeListen so we maintain an additional
			 * refcnt and free it up when nccl_net_ofi_closeRecv is
			 * called.
			 */
			pthread_mutex_lock(&nccl_ofi_lock);
			nccl_ofi_comp->refcnt++;
			pthread_mutex_unlock(&nccl_ofi_lock);

			/* Prepare receive request to accept connections */
			req = prepare_recv_conn(lComm);
			if (req == NULL) {
				put_nccl_ofi_comp(nccl_ofi_comp, dev);
				return ncclSystemError;
			}

			comm_state->stage = COMM_RECV_CONN;

		case COMM_RECV_CONN:

			/* Allocate memory for peer address for the first time ONLY */
			if (conn_info == NULL) {
				conn_info = calloc(1, sizeof(nccl_ofi_connection_info_t));
			}

			/* Post a receive message to receive peer connections */
			rc = post_recv_conn(lComm, nccl_ofi_comp, conn_info, sizeof(nccl_ofi_connection_info_t), req);
			if (rc == -FI_EAGAIN) {
				/* Save recv request and buffer address for retry */
				comm_state->req = req;
				lComm->buffer = conn_info;
				return ncclSuccess;
			} else if (rc != 0) {
				free(req);
				free(conn_info);
				lComm->buffer = NULL;
				put_nccl_ofi_comp(nccl_ofi_comp, dev);
				return ncclSystemError;
			}

			comm_state->stage = COMM_REQ_PENDING_COMP;

		case COMM_REQ_PENDING_COMP:

			/* Progress NCCL OFI engine so that connection is accepted */
			ret = nccl_ofi_progress(nccl_ofi_comp);
			if (OFI_UNLIKELY(ret != 0)) {
				free(req);
				put_nccl_ofi_comp(nccl_ofi_comp, dev);
				return ncclSystemError;
			}

			if (lComm->accepted != true) {
				/* Save recv request and buffer to retest completion */
				comm_state->req = req;
				lComm->buffer = conn_info;
				return ncclSuccess;
			}

			if (conn_info->connect_to_self) {
				NCCL_OFI_TRACE(NCCL_NET, "Accept from self; cleaning up");
				if (conn_info->req->state != NCCL_OFI_REQ_COMPLETED) {
					lComm->buffer = conn_info;
					return ncclSuccess;
				}
			}

			/* Done processing the request so free it */
			free(req);
			comm_state->stage = COMM_CONNECTED;

			break;

		case COMM_SEND_CONN:
		case COMM_CONNECTED:
		default:
			NCCL_OFI_WARN("Invalid state of receive communicator object: %d",
				      stage);
			/* TODO put ofi_comp here? */
			return ncclSystemError;
	}

	/* Prepare receive communicator object for the received peer connection */
	rComm = prepare_recv_comm(lComm, conn_info->ep_name);
	if (OFI_UNLIKELY(rComm == NULL)) {
		put_nccl_ofi_comp(nccl_ofi_comp, dev);
		return ncclSystemError;
	}

	free(conn_info);

	comm_state->comm = rComm;
	*recvComm = rComm;

	return ret;
}

#if HAVE_NEURON
ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, size_t size, int type,
#elif HAVE_CUDA
ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, int size, int type,
#endif
			      void **mhandle)
{
	struct fid_mr *mr_handle = NULL;
	ncclResult_t ret = ncclSuccess;

	ofiComm_t *ofi_comm = (ofiComm_t *)comm;

	/* Validate Comm */
	if (OFI_UNLIKELY(ofi_comm == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid Comm object provided");
		goto exit;
	}

	/* Validate type of buffer */
	switch (type) {
	case NCCL_PTR_HOST:
#if HAVE_CUDA
	case NCCL_PTR_CUDA:
#endif
#if HAVE_NEURON
	case NCCL_PTR_NEURON:
#endif
		break;
	default:
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid buffer type provided: %d", type);
		goto exit;
	};

	ret = register_mr_buffers(ofi_comm, data, size, type, &mr_handle);

exit:
	*mhandle = (void *)mr_handle;
	return ret;
}

ncclResult_t nccl_net_ofi_deregMr(void *comm, void *mhandle)
{
	ncclResult_t ret = ncclSuccess;
	int rc;
	struct fid_mr *mr_handle = (struct fid_mr *)mhandle;

	/* Validate Comm */
	if (OFI_UNLIKELY(comm == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid Comm object provided");
		goto exit;
	}

	if (OFI_LIKELY(mr_handle == NULL)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Null MR handle provided. Skipping deregisteration.");
		goto exit;
	}

	if (!prov_key_mr) {
		uint64_t key = fi_mr_key(mr_handle);
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Error retrieving MR key, leaking key");
		} else {
			ret = free_mr_key(((ofiComm_t *)comm)->dev, key);
			if (OFI_UNLIKELY(ret != ncclSuccess)) {
				NCCL_OFI_WARN("Error freeing MR key %"PRIu64", leaking key", key);
			}
		}
	}

	rc = fi_close((fid_t)mr_handle);
	if (OFI_UNLIKELY(rc != 0)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
				rc, fi_strerror(-rc));
	}

exit:
	return ret;
}

ncclResult_t nccl_net_ofi_regMrDmaBuf(void* comm, void* data, size_t size,
				    int type, uint64_t offset,
				    int fd, void** mhandle)
{
	return ncclInternalError;
}

ncclResult_t nccl_net_ofi_isend(void *sendComm, void* data, int size,
			      int tag, void *mhandle, void** request)
{
	ncclResult_t ret = ncclSuccess;
	ssize_t rc = 0;
	nccl_ofi_req_t *req = NULL;
	sendComm_t *sComm = (sendComm_t *)sendComm;
	nccl_ofi_t *nccl_ofi_comp = NULL;
	void *desc = NULL;

	/* Validate sendComm */
	if (OFI_UNLIKELY(sComm == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid sendComm provided");
		goto error;
	}

	/* Support only max_requests inflight requests. */
	if (OFI_UNLIKELY(sComm->num_inflight_reqs == max_requests)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      max_requests);
		goto error;
	}

	/*
	 * TODO: Use NCCL provided tags when using grouped receives aka
	 * props->maxRecvs > 1.
	 */

	/* Allocate NCCL OFI request */
	req = allocate_nccl_ofi_request(sComm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			     sComm->dev);
		goto error;
	}

	req->sComm = sComm;
	req->dev = sComm->dev;
	req->direction = NCCL_OFI_SEND;

	nccl_ofi_comp = sComm->baseComm.ofi_comp;
	if (OFI_UNLIKELY(nccl_ofi_comp == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("NCCL OFI component for dev %d is not initialised",
			     sComm->dev);
		goto error;
	}

	if (mhandle != NULL)
		desc = fi_mr_desc(mhandle);

	NCCL_OFI_TRACE_SEND(req->dev, size, req, request, &req->ctx);

	/*
	 * Try sending data to remote EP; Return NULL request
	 * if not able to send.
	 */
	rc = fi_tsend(sComm->local_ep, data, size, desc,
		      sComm->remote_ep, sComm->tag, &req->ctx);
	if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
		/* Make progress for next try */
		ret = nccl_ofi_progress(nccl_ofi_comp);
		/* Return NULL request */
		*request = NULL;
		goto error;
	}
	else if (OFI_UNLIKELY(rc != 0)) {
		NCCL_OFI_WARN("Could not send request for device %d. RC: %zd",
			     sComm->dev, rc);
		ret = ncclSystemError;
		goto error;
	}

	sComm->num_inflight_reqs++;

	/* Return request to NCCL */
	*request = req;

	goto exit;

error:
	if (req)
		free_nccl_ofi_req(req, false);
exit:
	return ret;
}

ncclResult_t nccl_net_ofi_irecv(void* recvComm, int n, void** buffers, int* sizes,
			      int *tags, void** mhandles, void** request)
{
	ncclResult_t ret = ncclSuccess;
	ssize_t rc = 0;
	nccl_ofi_req_t *req = NULL;
	recvComm_t *rComm = (recvComm_t *)recvComm;

	/* Validate recvComm */
	if (OFI_UNLIKELY(rComm == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid recvComm provided");
		goto error;
	}

	/* Support only max_requests inflight requests. */
	if (OFI_UNLIKELY(rComm->num_inflight_reqs == max_requests)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      max_requests);
		goto error;
	}

	/* Allocate NCCL OFI request */
	req = allocate_nccl_ofi_request(rComm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			     rComm->dev);
		goto error;
	}

	nccl_ofi_t *nccl_ofi_comp = rComm->baseComm.ofi_comp;
	if (OFI_UNLIKELY(nccl_ofi_comp == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("NCCL OFI component for dev %d is not initialised",
			     rComm->dev);
		goto error;
	}

	/* Progress NCCL OFI */
	ret = nccl_ofi_progress(nccl_ofi_comp);
	if (OFI_UNLIKELY(ret != 0))
		goto error;

	req->rComm = rComm;
	req->dev = rComm->dev;
	req->direction = NCCL_OFI_RECV;

	req->num_recvs = n;

	if (OFI_UNLIKELY(mhandles == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Memory handles array is NULL");
		goto error;
	}

	/* Currently, plugin doesn't support grouped receives */
	assert(n <= NCCL_OFI_MAX_RECVS);
	for (int recv_n = 0; recv_n < n; recv_n++) {
		void *desc = NULL;

		if (mhandles[recv_n] != NULL) {
			desc = fi_mr_desc(mhandles[recv_n]);
		}

        NCCL_OFI_TRACE_RECV(rComm->dev, rComm->tag, sizes[recv_n], req, request, &req->ctx);

		/*
		 * TODO: Use NCCL provided tags when plugin supports grouped
		 * receives aka props->maxRecvs > 1.
		 */

		/* Try posting buffer to local EP */
		rc = fi_trecv(rComm->local_ep, buffers[recv_n], sizes[recv_n],
			      desc, FI_ADDR_UNSPEC, rComm->tag, 0, &req->ctx);
		if (rc == -FI_EAGAIN) {
			/* Return NULL request */
			*request = NULL;
			goto error;
		}
		else if (rc != 0) {
			NCCL_OFI_WARN("Unable to post receive buffer for dev %d. RC: %zd, ERROR: %s",
					rComm->dev, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto error;
		}

	}

	rComm->num_inflight_reqs++;

	/* Return request to NCCL */
	*request = req;

	goto exit;

error:
	if (req)
		free_nccl_ofi_req(req, false);
exit:
	return ret;
}

ncclResult_t nccl_net_ofi_test(void* request, int* done, int* size)
{
	ncclResult_t ret = ncclSuccess;

	/* Check if request is valid */
	if (OFI_UNLIKELY(request == NULL)) {
		ret = ncclInternalError;
		goto exit;
	}

	nccl_ofi_req_t *req = (nccl_ofi_req_t *)request;
	nccl_ofi_t *nccl_ofi_comp = req->bComm->ofi_comp;
	if (OFI_UNLIKELY(nccl_ofi_comp == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("NCCL OFI component for dev %d is uninitialised",
			      req->dev);
		goto exit;
	}

	/* Process more completions unless the current request is completed */
	if (req->state != NCCL_OFI_REQ_COMPLETED) {
		ret = nccl_ofi_progress(nccl_ofi_comp);
		if (OFI_UNLIKELY(ret != 0))
			goto exit;
	}

	/* Determine whether the request has finished and free if done */
	if (OFI_LIKELY(req->state == NCCL_OFI_REQ_COMPLETED ||
		       req->state == NCCL_OFI_REQ_ERROR)) {
		__compiler_barrier();
		if (size)
			*size = req->size;
		/* Mark as done */
		*done = 1;

		if (OFI_UNLIKELY(req->state == NCCL_OFI_REQ_ERROR))
			ret = ncclSystemError;
		free_nccl_ofi_req(req, true);
	}
	else {
		*done = 0;
	}

exit:
	return ret;
}

ncclResult_t nccl_net_ofi_iflush(void* recvComm, int n, void** buffers, int* sizes,
			       void** mhandles, void** request)
{
	ncclResult_t ret = ncclSuccess;
	recvComm_t *rComm = (recvComm_t *)recvComm;
	nccl_ofi_req_t *req = NULL;
	ssize_t rc = 0;
	uint64_t cuda_key = 0ULL;
	struct fid_mr *mr_handle = NULL;
	void *data = NULL;
	void *flush_mr_desc = NULL;

	if (ofi_nccl_gdr_flush_disable() || !support_gdr)
		goto exit;

#if CUDART_VERSION >= 11030
	if (cuda_flush) {
		cudaError_t cuda_ret = cudaDeviceFlushGPUDirectRDMAWrites(
						cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
						cudaFlushGPUDirectRDMAWritesToOwner);

		if (cuda_ret != cudaSuccess) {
			ret = ncclUnhandledCudaError;
			NCCL_OFI_WARN("Error performing CUDA GDR flush");
			goto exit;
		}

		goto exit;
	}
#endif

	/* Validate recvComm */
	if (OFI_UNLIKELY(rComm == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid recvComm provided");
		goto exit;
	}

	/* Plugin only supports one receive per request */
	assert(n <= NCCL_OFI_MAX_RECVS);

	/*
	 * Find the non-zero request for which we will issue flush.
	 * A single operation can flush all request at once.
	 */
	int flush_n = -1;
	for (int recv_n = 0; recv_n < n; recv_n++) {
		if (sizes[recv_n] != 0) {
			flush_n = recv_n;
			break;
		}
	}

	if (flush_n == -1) {
		/*
		 * Flush is an expensive operation. So, don't send fi_read for
		 * 0-sized messages. Since, NCCL issues flush for every irecv(),
		 * we guarantee to sync data to GPU even without it.
		 */
		goto exit;
	}

	if (mhandles && mhandles[flush_n])
		mr_handle = (struct fid_mr *)mhandles[flush_n];

	data = buffers[flush_n];

	/* Support only max_requests inflight requests. */
	if (OFI_UNLIKELY(rComm->num_inflight_reqs == max_requests)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      max_requests);
		goto exit;
	}

	/* Allocate NCCL OFI request */
	req = allocate_nccl_ofi_request(rComm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			     rComm->dev);
		goto exit;
	}

	req->rComm = rComm;
	req->dev = rComm->dev;
	req->direction = NCCL_OFI_RECV;

	if (rComm->flush_buff.mr_handle != NULL) {
		/* Not checking for NULL flush_mr_desc as fi_mr_desc()
		 * returns valid descriptors by valid handles */
		flush_mr_desc = fi_mr_desc(rComm->flush_buff.mr_handle);
	}

	if (mr_handle != NULL) {
		/* Extract remote key */
		cuda_key = fi_mr_key(mr_handle);
		if (OFI_UNLIKELY(cuda_key == FI_KEY_NOTAVAIL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Memory registration may not have completed.");
			goto error;
		}
	}

	NCCL_OFI_TRACE_FLUSH(req, request, &req->ctx);

	/* Issue RDMA read */
	do {
		rc = fi_read(rComm->local_ep, rComm->flush_buff.host_buffer,
			     rComm->flush_buff.size,
			     flush_mr_desc,
			     rComm->local_ep_addr,
			     (uint64_t)(virt_addr_mr ? data : 0),
			     cuda_key, &req->ctx);
		if (rc == 0) {
			break;
		}
		else if (rc == -FI_EAGAIN) {
			/*
			 * Process completions so that you have enough
			 * resources for issuing fi_read
			 */
			ret = nccl_ofi_progress(rComm->baseComm.ofi_comp);
			if (OFI_UNLIKELY(ret != ncclSuccess))
				goto error;
		}
		else {
			NCCL_OFI_WARN("Unable to issue read operation for dev %d. RC: %zd, ERROR: %s",
					rComm->dev, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto error;
		}
	} while (true);

	rComm->num_inflight_reqs++;

	*request = req;

	return ret;

error:
	if (req)
		free_nccl_ofi_req(req, false);
exit:
	*request = NULL;
	return ret;
}

ncclResult_t nccl_net_ofi_closeSend(void *sendComm)
{
	sendComm_t *sComm = (sendComm_t *)sendComm;
	int dev;
	ncclResult_t ret = ncclSuccess;

	if (OFI_UNLIKELY(sendComm == NULL)) {
		ret = ncclInternalError;
		goto exit;
	}

	dev = sComm->dev;

	free_ofi_fl(sComm->nccl_ofi_reqs_fl);
	put_nccl_ofi_comp(sComm->baseComm.ofi_comp, dev);
	free(sendComm);
exit:
	return ret;
}

ncclResult_t nccl_net_ofi_closeRecv(void *recvComm)
{
	recvComm_t *rComm = (recvComm_t *)recvComm;
	int dev, rc;
	ncclResult_t ret = ncclSuccess;
	struct fid_mr *mr_handle = NULL;

	if (OFI_UNLIKELY(recvComm == NULL)) {
		ret = ncclInternalError;
		goto exit;
	}

	dev = rComm->dev;

	if (!ofi_nccl_gdr_flush_disable() && support_gdr && !cuda_flush) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "De-registering buffer for flush operations");
		/* Deregister Flush buffer memory region */
		mr_handle = (struct fid_mr *)rComm->flush_buff.mr_handle;
		if (mr_handle) {
			rc = fi_close((fid_t)mr_handle);
			if (OFI_UNLIKELY(rc != 0)) {
				ret = ncclSystemError;
				NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
						rc, fi_strerror(-rc));
				goto exit;
			}
		}
		if (munmap(rComm->flush_buff.host_buffer, sysconf(_SC_PAGESIZE))) {
			NCCL_OFI_WARN("Unable to unmap flush buffer (%d %s)", errno, strerror(errno));
		}
		rComm->flush_buff.host_buffer = MAP_FAILED;
	}

	free_ofi_fl(rComm->nccl_ofi_reqs_fl);
	put_nccl_ofi_comp(rComm->baseComm.ofi_comp, dev);
	free(recvComm);
exit:
	return ret;
}

ncclResult_t nccl_net_ofi_closeListen(void *listenComm)
{
	listenComm_t *lComm = (listenComm_t *)listenComm;
	int dev;
	ncclResult_t ret = ncclSuccess;

	if (OFI_UNLIKELY(listenComm == NULL)) {
		ret = ncclInternalError;
		goto exit;
	}

	dev = lComm->dev;

	put_nccl_ofi_comp(lComm->baseComm.ofi_comp, dev);
	free(listenComm);
exit:
	return ret;
}
