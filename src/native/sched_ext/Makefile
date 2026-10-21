# SPDX-License-Identifier: GPL-2.0

KERNEL_SRC ?= /opt/kernel-src/linux-6.12.y
EXPECTED_KERNEL_COMMIT := 296aabce459470a4c1b68ffd0c0c0920e563aaad

CC := gcc
CLANG ?= clang
GIT ?= git

CURDIR := $(abspath .)
SCX_DIR := $(abspath $(KERNEL_SRC)/tools/sched_ext)
SCX_BUILD := $(SCX_DIR)/build
BUILD_DIR := $(CURDIR)/build
OBJ_DIR := $(BUILD_DIR)/obj
INCLUDE_DIR := $(BUILD_DIR)/include
BIN_DIR := $(BUILD_DIR)/bin

BPFTOOL := $(SCX_BUILD)/sbin/bpftool
LIBBPF := $(SCX_BUILD)/obj/libbpf/libbpf.a
VMLINUX_H := $(SCX_BUILD)/include/vmlinux.h
BPF_OBJ := $(OBJ_DIR)/agentos_scx.bpf.o
SKEL := $(INCLUDE_DIR)/agentos_scx.bpf.skel.h
LOADER_OBJ := $(OBJ_DIR)/agentos_scx.o
LOADER := $(BIN_DIR)/agentos_scx
WORKER := $(BIN_DIR)/agentos_scx_worker
BENCH_WORKER := $(BIN_DIR)/agentos_scx_bench_worker
SIMPLE_PARTIAL_OBJ := $(OBJ_DIR)/scx_simple_partial.o
SIMPLE_PARTIAL := $(BIN_DIR)/scx_simple_partial
SIMPLE_SKEL := $(SCX_BUILD)/include/scx_simple.bpf.skel.h

SYS_INCLUDES := $(shell $(CLANG) -v -E - </dev/null 2>&1 | \
	sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

BPF_CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_x86 -mlittle-endian -mcpu=v3 \
	-Wall -Werror -Wno-compare-distinct-pointer-types \
	-I$(CURDIR)/include \
	-I$(SCX_DIR)/include \
	-I$(SCX_DIR)/include/bpf-compat \
	-I$(SCX_BUILD)/include \
	-I$(KERNEL_SRC)/tools/include/uapi \
	-I$(KERNEL_SRC)/include \
	$(SYS_INCLUDES)

CFLAGS := -g -O2 -Wall -Wextra -Werror -Wno-error=sign-compare -pthread \
	-I$(CURDIR)/include \
	-I$(INCLUDE_DIR) \
	-I$(SCX_BUILD)/include \
	-I$(KERNEL_SRC)/include/generated \
	-I$(KERNEL_SRC)/tools/lib \
	-I$(KERNEL_SRC)/tools/include \
	-I$(KERNEL_SRC)/tools/include/uapi \
	-I$(SCX_DIR)/include
LDFLAGS := -lelf -lz -lpthread

.PHONY: all clean print-config check-kernel-source upstream-tools

all: $(LOADER) $(WORKER) $(BENCH_WORKER) $(SIMPLE_PARTIAL)

check-kernel-source:
	@if [ ! -d "$(KERNEL_SRC)/.git" ]; then \
		echo "KERNEL_SRC is not a git checkout: $(KERNEL_SRC)" >&2; exit 1; \
	fi
	@actual="$$($(GIT) -C "$(KERNEL_SRC)" rev-parse HEAD 2>/dev/null)"; \
	if [ "$$actual" != "$(EXPECTED_KERNEL_COMMIT)" ]; then \
		echo "KERNEL_SRC commit must be $(EXPECTED_KERNEL_COMMIT), got $$actual" >&2; exit 1; \
	fi
	@test -f "$(KERNEL_SRC)/vmlinux" || \
		{ echo "KERNEL_SRC/vmlinux is missing; build the pinned kernel first" >&2; exit 1; }

upstream-tools: check-kernel-source
	$(MAKE) -C "$(SCX_DIR)" all

$(BPFTOOL) $(LIBBPF) $(VMLINUX_H): upstream-tools
	@test -f $@ || { echo "upstream build did not create $@" >&2; exit 1; }

$(OBJ_DIR) $(INCLUDE_DIR) $(BIN_DIR):
	mkdir -p $@

$(BPF_OBJ): agentos_scx.bpf.c include/agentos_scx.h $(VMLINUX_H) | $(OBJ_DIR)
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

$(SKEL): $(BPF_OBJ) $(BPFTOOL) | $(INCLUDE_DIR)
	$(BPFTOOL) gen object $(OBJ_DIR)/agentos_scx.bpf.linked1.o $<
	$(BPFTOOL) gen object $(OBJ_DIR)/agentos_scx.bpf.linked2.o $(OBJ_DIR)/agentos_scx.bpf.linked1.o
	$(BPFTOOL) gen object $(OBJ_DIR)/agentos_scx.bpf.linked3.o $(OBJ_DIR)/agentos_scx.bpf.linked2.o
	diff $(OBJ_DIR)/agentos_scx.bpf.linked2.o $(OBJ_DIR)/agentos_scx.bpf.linked3.o
	$(BPFTOOL) gen skeleton $(OBJ_DIR)/agentos_scx.bpf.linked3.o name agentos_scx > $@

$(LOADER_OBJ): agentos_scx.c include/agentos_scx.h $(SKEL) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LOADER): $(LOADER_OBJ) $(LIBBPF) | $(BIN_DIR)
	$(CC) -o $@ $(LOADER_OBJ) $(LIBBPF) $(LDFLAGS)

$(WORKER): agentos_scx_worker.c | $(BIN_DIR)
	$(CC) -g -O2 -Wall -Wextra -Werror $< -o $@

$(BENCH_WORKER): agentos_scx_bench_worker.c | $(BIN_DIR)
	$(CC) -g -O2 -Wall -Wextra -Werror $< -o $@

$(SIMPLE_PARTIAL_OBJ): scx_simple_partial.c $(SIMPLE_SKEL) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(SIMPLE_PARTIAL): $(SIMPLE_PARTIAL_OBJ) $(LIBBPF) | $(BIN_DIR)
	$(CC) -o $@ $(SIMPLE_PARTIAL_OBJ) $(LIBBPF) $(LDFLAGS)

print-config:
	@echo "KERNEL_SRC=$(KERNEL_SRC)"
	@echo "EXPECTED_KERNEL_COMMIT=$(EXPECTED_KERNEL_COMMIT)"
	@echo "SCX_DIR=$(SCX_DIR)"
	@echo "SCX_BUILD=$(SCX_BUILD)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
	@echo "CC=$(CC)"
	@echo "CLANG=$(CLANG)"

clean:
	rm -rf $(BUILD_DIR)
