#
# Copyright (c) 2018-2019, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CFLAGS = -g -O3 -Wall -fPIC -Wno-sign-compare
AM_CFLAGS += -I$(top_srcdir)/include
AM_LDFLAGS = -lcudart

lib_LTLIBRARIES = libnccl-net.la
libnccl_net_la_SOURCES =

#libnccl_net_la_LIBADD = libv4.la libv5.la
libnccl_net_la_LIBADD = $(PLUGIN_VERSION_LIBS)
libnccl_net_la_DEPENDENCIES = $(PLUGIN_VERSION_LIBS)

EXTRA_LTLIBRARIES =

libv2_la_SOURCES = nccl_ofi_net.c
libv2_la_CFLAGS = $(AM_CFLAGS) -DPLUGIN_VERSION=2
EXTRA_LTLIBRARIES += libv2.la

libv3_la_SOURCES = nccl_ofi_net.c
libv3_la_CFLAGS = $(AM_CFLAGS) -DPLUGIN_VERSION=3
EXTRA_LTLIBRARIES += libv3.la

libv4_la_SOURCES = nccl_ofi_net.c
libv4_la_CFLAGS = $(AM_CFLAGS) -DPLUGIN_VERSION=4
EXTRA_LTLIBRARIES += libv4.la

libv5_la_SOURCES = nccl_ofi_net.c
libv5_la_CFLAGS = $(AM_CFLAGS) -DPLUGIN_VERSION=5
EXTRA_LTLIBRARIES += libv5.la

libv6_la_SOURCES = nccl_ofi_net.c
libv6_la_CFLAGS = $(AM_CFLAGS) -DPLUGIN_VERSION=6
EXTRA_LTLIBRARIES += libv6.la
