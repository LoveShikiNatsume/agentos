/* SPDX-License-Identifier: GPL-2.0 */
#define _GNU_SOURCE

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <bpf/libbpf.h>
#include <scx/common.h>

#include "scx_simple.bpf.skel.h"

static volatile sig_atomic_t exit_requested;

static void signal_handler(int signo)
{
	exit_requested = signo;
}

int main(void)
{
	struct scx_simple *skel = NULL;
	struct bpf_link *link = NULL;
	struct sigaction action = {.sa_handler = signal_handler};
	struct timespec delay = {.tv_sec = 0, .tv_nsec = 100 * 1000 * 1000};
	bool unexpected_exit;
	int return_code = 1;

	if (geteuid() != 0) {
		fprintf(stderr, "scx_simple_partial must run as root\n");
		return 1;
	}
	sigemptyset(&action.sa_mask);
	sigaction(SIGINT, &action, NULL);
	sigaction(SIGTERM, &action, NULL);

	skel = scx_simple__open();
	if (!skel) {
		fprintf(stderr, "failed to open scx_simple skeleton\n");
		goto out;
	}
	skel->struct_ops.simple_ops->flags |= SCX_OPS_SWITCH_PARTIAL;
	skel->struct_ops.simple_ops->hotplug_seq = scx_hotplug_seq();
	UEI_SET_SIZE(skel, simple_ops, uei);
	if (scx_simple__load(skel)) {
		fprintf(stderr, "failed to load scx_simple skeleton\n");
		goto out;
	}
#if LIBBPF_MAJOR_VERSION > 1 || \
	(LIBBPF_MAJOR_VERSION == 1 && LIBBPF_MINOR_VERSION >= 5)
	bpf_map__set_autoattach(skel->maps.simple_ops, false);
#endif
	if (scx_simple__attach(skel)) {
		fprintf(stderr, "failed to attach scx_simple programs\n");
		goto out;
	}
	link = bpf_map__attach_struct_ops(skel->maps.simple_ops);
	if (!link) {
		fprintf(stderr, "failed to attach scx_simple struct_ops\n");
		goto out;
	}

	while (!exit_requested && !UEI_EXITED(skel, uei))
		nanosleep(&delay, NULL);
	unexpected_exit = !exit_requested && UEI_EXITED(skel, uei);
	return_code = unexpected_exit ? 1 : 0;

out:
	bpf_link__destroy(link);
	scx_simple__destroy(skel);
	return return_code;
}
