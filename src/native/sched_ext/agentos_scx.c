/* SPDX-License-Identifier: GPL-2.0 */
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <scx/common.h>

#include "agentos_scx.h"
#include "agentos_scx.bpf.skel.h"

static volatile sig_atomic_t exit_requested;
static bool verbose;

static const char *const stat_names[AGENTOS_SCX_STAT_NR] = {
	[AGENTOS_SCX_STAT_SELECT_CPU] = "select_cpu",
	[AGENTOS_SCX_STAT_DIRECT_LOCAL] = "direct_local",
	[AGENTOS_SCX_STAT_ENQUEUE] = "enqueue",
	[AGENTOS_SCX_STAT_DISPATCH] = "dispatch",
	[AGENTOS_SCX_STAT_RUNNING] = "running",
	[AGENTOS_SCX_STAT_STOPPING] = "stopping",
	[AGENTOS_SCX_STAT_UNKNOWN_POLICY] = "unknown_policy",
	[AGENTOS_SCX_STAT_POLICY_HIT] = "policy_hit",
	[AGENTOS_SCX_STAT_ENQUEUE_LATENCY] = "enqueue_latency",
	[AGENTOS_SCX_STAT_ENQUEUE_NORMAL] = "enqueue_normal",
	[AGENTOS_SCX_STAT_ENQUEUE_BATCH] = "enqueue_batch",
	[AGENTOS_SCX_STAT_DISPATCH_LATENCY] = "dispatch_latency",
	[AGENTOS_SCX_STAT_DISPATCH_NORMAL] = "dispatch_normal",
	[AGENTOS_SCX_STAT_DISPATCH_BATCH] = "dispatch_batch",
	[AGENTOS_SCX_STAT_DIRECT_LATENCY] = "direct_latency",
	[AGENTOS_SCX_STAT_DIRECT_NORMAL] = "direct_normal",
	[AGENTOS_SCX_STAT_DIRECT_BATCH] = "direct_batch",
	[AGENTOS_SCX_STAT_WAIT_TOTAL_LATENCY_NS] = "wait_total_latency_ns",
	[AGENTOS_SCX_STAT_WAIT_TOTAL_NORMAL_NS] = "wait_total_normal_ns",
	[AGENTOS_SCX_STAT_WAIT_TOTAL_BATCH_NS] = "wait_total_batch_ns",
	[AGENTOS_SCX_STAT_WAIT_MAX_LATENCY_NS] = "wait_max_latency_ns",
	[AGENTOS_SCX_STAT_WAIT_MAX_NORMAL_NS] = "wait_max_normal_ns",
	[AGENTOS_SCX_STAT_WAIT_MAX_BATCH_NS] = "wait_max_batch_ns",
	[AGENTOS_SCX_STAT_RUNTIME_LATENCY_NS] = "runtime_latency_ns",
	[AGENTOS_SCX_STAT_RUNTIME_NORMAL_NS] = "runtime_normal_ns",
	[AGENTOS_SCX_STAT_RUNTIME_BATCH_NS] = "runtime_batch_ns",
	[AGENTOS_SCX_STAT_MIGRATION] = "migration",
	[AGENTOS_SCX_STAT_FALLBACK] = "fallback",
	[AGENTOS_SCX_STAT_FALLBACK_LOOKUP] = "fallback_lookup",
	[AGENTOS_SCX_STAT_FALLBACK_VERSION] = "fallback_version",
	[AGENTOS_SCX_STAT_FALLBACK_RANGE] = "fallback_range",
	[AGENTOS_SCX_STAT_FALLBACK_GENERATION] = "fallback_generation",
	[AGENTOS_SCX_STAT_FALLBACK_HEARTBEAT] = "fallback_heartbeat",
	[AGENTOS_SCX_STAT_FALLBACK_POLICY_STALE] = "fallback_policy_stale",
	[AGENTOS_SCX_STAT_FALLBACK_STATE] = "fallback_state",
	[AGENTOS_SCX_STAT_BATCH_BOOST] = "batch_boost",
	[AGENTOS_SCX_STAT_PREEMPT_REQUESTED] = "preempt_requested",
	[AGENTOS_SCX_STAT_PREEMPT_SUPPRESSED] = "preempt_suppressed",
};

struct pinned_map {
	struct bpf_map *map;
	const char *name;
};

static void signal_handler(int signo)
{
	exit_requested = signo;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	if (level == LIBBPF_DEBUG && !verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void json_string(const char *value)
{
	const unsigned char *cursor = (const unsigned char *)(value ? value : "");

	putchar('"');
	for (; *cursor; cursor++) {
		switch (*cursor) {
		case '\\':
			fputs("\\\\", stdout);
			break;
		case '"':
			fputs("\\\"", stdout);
			break;
		case '\n':
			fputs("\\n", stdout);
			break;
		case '\r':
			fputs("\\r", stdout);
			break;
		case '\t':
			fputs("\\t", stdout);
			break;
		default:
			if (*cursor < 0x20)
				printf("\\u%04x", *cursor);
			else
				putchar(*cursor);
		}
	}
	putchar('"');
}

static int read_text(const char *path, char *buffer, size_t size)
{
	FILE *file;
	size_t length;

	file = fopen(path, "re");
	if (!file)
		return -errno;
	if (!fgets(buffer, size, file)) {
		int saved = ferror(file) ? errno : EIO;

		fclose(file);
		return -saved;
	}
	fclose(file);
	length = strlen(buffer);
	while (length && (buffer[length - 1] == '\n' || buffer[length - 1] == '\r'))
		buffer[--length] = '\0';
	return 0;
}

static int wait_for_value(const char *path, const char *expected,
			  unsigned int timeout_ms)
{
	struct timespec delay = {.tv_sec = 0, .tv_nsec = 20 * 1000 * 1000};
	char value[128];
	unsigned int elapsed = 0;

	while (elapsed <= timeout_ms) {
		if (!read_text(path, value, sizeof(value)) && !strcmp(value, expected))
			return 0;
		nanosleep(&delay, NULL);
		elapsed += 20;
	}
	return -ETIMEDOUT;
}

static int verify_enabled(void)
{
	if (wait_for_value("/sys/kernel/sched_ext/state", "enabled", 2000))
		return -1;
	if (wait_for_value("/sys/kernel/sched_ext/switch_all", "0", 2000))
		return -1;
	if (wait_for_value("/sys/kernel/sched_ext/root/ops", AGENTOS_SCX_NAME, 2000))
		return -1;
	return 0;
}

static int ensure_pin_directory(void)
{
	if (mkdir("/run/agentos", 0755) && errno != EEXIST)
		return -errno;
	if (mkdir("/sys/fs/bpf/agentos", 0755) && errno != EEXIST)
		return -errno;
	if (mkdir(AGENTOS_SCX_PIN_DIR, 0755) && errno != EEXIST)
		return -errno;
	return 0;
}

static int update_loader_heartbeat(struct agentos_scx *skel)
{
	struct agentos_global_config_v1 config;
	struct timespec now;
	uint32_t key = 0;
	int lock_fd;
	int error = 0;

	lock_fd = open(AGENTOS_SCX_CONTROL_LOCK, O_CREAT | O_CLOEXEC | O_RDWR,
		       0644);
	if (lock_fd < 0)
		return -errno;
	if (flock(lock_fd, LOCK_EX)) {
		error = -errno;
		goto out;
	}
	if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.global_config_map),
				&key, &config)) {
		error = -errno;
		goto unlock;
	}
	if (clock_gettime(CLOCK_MONOTONIC, &now)) {
		error = -errno;
		goto unlock;
	}
	config.heartbeat_ns = (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec;
	config.loader_pid = getpid();
	if (bpf_map_update_elem(bpf_map__fd(skel->maps.global_config_map),
				&key, &config, BPF_ANY))
		error = -errno;
unlock:
	flock(lock_fd, LOCK_UN);
out:
	close(lock_fd);
	return error;
}

static uint64_t monotonic_ns(void)
{
	struct timespec now;

	if (clock_gettime(CLOCK_MONOTONIC, &now))
		return 0;
	return (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec;
}

static int pin_maps(struct pinned_map *maps, size_t count)
{
	size_t index;

	if (ensure_pin_directory())
		return -1;
	for (index = 0; index < count; index++) {
		char path[256];

		snprintf(path, sizeof(path), "%s/%s", AGENTOS_SCX_PIN_DIR,
			 maps[index].name);
		if (bpf_map__pin(maps[index].map, path)) {
			while (index-- > 0) {
				snprintf(path, sizeof(path), "%s/%s",
					 AGENTOS_SCX_PIN_DIR, maps[index].name);
				bpf_map__unpin(maps[index].map, path);
			}
			return -1;
		}
	}
	return 0;
}

static void unpin_maps(struct pinned_map *maps, size_t count)
{
	char path[256];

	while (count-- > 0) {
		snprintf(path, sizeof(path), "%s/%s", AGENTOS_SCX_PIN_DIR,
			 maps[count].name);
		bpf_map__unpin(maps[count].map, path);
	}
}

static int initialize_global_config(struct agentos_scx *skel)
{
	struct timespec now;
	struct agentos_global_config_v1 config = {
		.schema_version = AGENTOS_SCX_POLICY_SCHEMA_VERSION,
		.loader_pid = getpid(),
		.reserved = AGENTOS_SCX_HEARTBEAT_TIMEOUT_SEC,
	};
	uint32_t key = 0;

	if (clock_gettime(CLOCK_MONOTONIC, &now))
		return -1;
	config.generation = (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec;
	config.heartbeat_ns = config.generation;
	return bpf_map_update_elem(
		bpf_map__fd(skel->maps.global_config_map), &key, &config, BPF_ANY
	);
}

static int read_stats(struct agentos_scx *skel, uint64_t *totals)
{
	int nr_cpus = libbpf_num_possible_cpus();
	uint64_t *values;
	uint32_t index;

	if (nr_cpus <= 0)
		return -EINVAL;
	values = calloc((size_t)nr_cpus, sizeof(*values));
	if (!values)
		return -ENOMEM;

	memset(totals, 0, sizeof(*totals) * AGENTOS_SCX_STAT_NR);
	for (index = 0; index < AGENTOS_SCX_STAT_NR; index++) {
		int cpu;

		memset(values, 0, (size_t)nr_cpus * sizeof(*values));
		if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.stats), &index, values)) {
			free(values);
			return -errno;
		}
		for (cpu = 0; cpu < nr_cpus; cpu++) {
			if (index == AGENTOS_SCX_STAT_WAIT_MAX_LATENCY_NS ||
			    index == AGENTOS_SCX_STAT_WAIT_MAX_NORMAL_NS ||
			    index == AGENTOS_SCX_STAT_WAIT_MAX_BATCH_NS) {
				if (totals[index] < values[cpu])
					totals[index] = values[cpu];
			} else {
				totals[index] += values[cpu];
			}
		}
	}
	free(values);
	return 0;
}

static void emit_error(const char *message)
{
	fputs("{\"event\":\"error\",\"message\":", stdout);
	json_string(message);
	fputs("}\n", stdout);
	fflush(stdout);
}

static void emit_stats(struct agentos_scx *skel, const char *event)
{
	uint64_t totals[AGENTOS_SCX_STAT_NR];
	uint64_t latency_runs;
	uint64_t normal_runs;
	uint64_t batch_runs;
	int index;

	if (read_stats(skel, totals)) {
		emit_error("failed to read stats map");
		return;
	}
	printf("{\"event\":\"%s\",\"stats\":{", event);
	for (index = 0; index < AGENTOS_SCX_STAT_NR; index++)
		printf("%s\"%s\":%llu", index ? "," : "", stat_names[index],
		       (unsigned long long)totals[index]);
	latency_runs = totals[AGENTOS_SCX_STAT_DISPATCH_LATENCY] +
		       totals[AGENTOS_SCX_STAT_DIRECT_LATENCY];
	normal_runs = totals[AGENTOS_SCX_STAT_DISPATCH_NORMAL] +
		      totals[AGENTOS_SCX_STAT_DIRECT_NORMAL];
	batch_runs = totals[AGENTOS_SCX_STAT_DISPATCH_BATCH] +
		     totals[AGENTOS_SCX_STAT_DIRECT_BATCH];
	fputs("},\"derived\":{", stdout);
	printf("\"wait_avg_latency_ns\":%llu,",
	       (unsigned long long)(
		       latency_runs ?
		       totals[AGENTOS_SCX_STAT_WAIT_TOTAL_LATENCY_NS] /
		       latency_runs : 0));
	printf("\"wait_avg_normal_ns\":%llu,",
	       (unsigned long long)(
		       normal_runs ?
		       totals[AGENTOS_SCX_STAT_WAIT_TOTAL_NORMAL_NS] /
		       normal_runs : 0));
	printf("\"wait_avg_batch_ns\":%llu",
	       (unsigned long long)(
		       batch_runs ?
		       totals[AGENTOS_SCX_STAT_WAIT_TOTAL_BATCH_NS] /
		       batch_runs : 0));
	fputs("}}\n", stdout);
	fflush(stdout);
}

static int parse_interval(const char *value, unsigned int *interval)
{
	char *end;
	unsigned long parsed;

	errno = 0;
	parsed = strtoul(value, &end, 10);
	if (errno || *end || parsed < 10 || parsed > 60000)
		return -1;
	*interval = (unsigned int)parsed;
	return 0;
}

static void usage(FILE *stream, const char *program)
{
	fprintf(stream,
		"Usage: %s [--stats-interval-ms N] [--quiet] [--verbose]\n",
		program);
}

int main(int argc, char **argv)
{
	static const struct option options[] = {
		{"stats-interval-ms", required_argument, NULL, 'i'},
		{"quiet", no_argument, NULL, 'q'},
		{"verbose", no_argument, NULL, 'v'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	struct agentos_scx *skel = NULL;
	struct bpf_link *link = NULL;
	struct pinned_map maps[4];
	bool maps_pinned = false;
	unsigned int interval_ms = 1000;
	bool quiet = false;
	bool unexpected_exit = false;
	bool heartbeat_failed = false;
	uint64_t next_stats_ns = 0;
	uint64_t next_heartbeat_ns = 0;
	int option;
	int return_code = 1;
	struct sigaction action = {.sa_handler = signal_handler};

	while ((option = getopt_long(argc, argv, "i:qvh", options, NULL)) != -1) {
		switch (option) {
		case 'i':
			if (parse_interval(optarg, &interval_ms)) {
				usage(stderr, argv[0]);
				return 2;
			}
			break;
		case 'q':
			quiet = true;
			break;
		case 'v':
			verbose = true;
			break;
		case 'h':
			usage(stdout, argv[0]);
			return 0;
		default:
			usage(stderr, argv[0]);
			return 2;
		}
	}

	if (geteuid() != 0) {
		emit_error("agentos_scx must run as root");
		return 1;
	}
	libbpf_set_print(libbpf_print_fn);
	sigemptyset(&action.sa_mask);
	sigaction(SIGINT, &action, NULL);
	sigaction(SIGTERM, &action, NULL);

	skel = agentos_scx__open();
	if (!skel) {
		emit_error("failed to open BPF skeleton");
		goto out;
	}
	skel->struct_ops.agentos_scx_ops->hotplug_seq = scx_hotplug_seq();
	UEI_SET_SIZE(skel, agentos_scx_ops, uei);
	if (agentos_scx__load(skel)) {
		emit_error("failed to load BPF skeleton");
		goto out;
	}
	maps[0] = (struct pinned_map){skel->maps.task_policy_map, "task_policy_map"};
	maps[1] = (struct pinned_map){skel->maps.task_state_map, "task_state_map"};
	maps[2] = (struct pinned_map){skel->maps.global_config_map, "global_config_map"};
	maps[3] = (struct pinned_map){skel->maps.stats, "stats_map"};
	if (initialize_global_config(skel)) {
		emit_error("failed to initialize global config map");
		goto out;
	}
	if (pin_maps(maps, 4)) {
		emit_error("failed to pin sched_ext maps");
		goto out;
	}
	maps_pinned = true;
#if LIBBPF_MAJOR_VERSION > 1 || \
	(LIBBPF_MAJOR_VERSION == 1 && LIBBPF_MINOR_VERSION >= 5)
	bpf_map__set_autoattach(skel->maps.agentos_scx_ops, false);
#endif
	if (agentos_scx__attach(skel)) {
		emit_error("failed to attach BPF programs");
		goto out;
	}
	link = bpf_map__attach_struct_ops(skel->maps.agentos_scx_ops);
	if (!link) {
		emit_error("failed to attach sched_ext struct_ops");
		goto out;
	}
	if (verify_enabled()) {
		emit_error("sched_ext sysfs did not report partial agentos_scx");
		goto out;
	}

	printf("{\"event\":\"ready\",\"scheduler\":\"%s\","
	       "\"partial\":true,\"pid\":%d,\"policy_schema_version\":%u,"
	       "\"map_pin_dir\":\"%s\"}\n",
	       AGENTOS_SCX_NAME, getpid(), AGENTOS_SCX_POLICY_SCHEMA_VERSION,
	       AGENTOS_SCX_PIN_DIR);
	fflush(stdout);
	next_stats_ns = monotonic_ns();

	while (!exit_requested && !UEI_EXITED(skel, uei)) {
		unsigned int heartbeat_delay_ms = interval_ms < 1000 ? interval_ms : 1000;
		struct timespec delay = {
			.tv_sec = heartbeat_delay_ms / 1000,
			.tv_nsec = (long)(heartbeat_delay_ms % 1000) * 1000 * 1000,
		};
		uint64_t now;

		now = monotonic_ns();
		if (now >= next_heartbeat_ns) {
			if (update_loader_heartbeat(skel)) {
				emit_error("failed to update loader heartbeat");
				heartbeat_failed = true;
				break;
			}
			next_heartbeat_ns = now + 1000000000ULL;
		}
		if (!quiet && now >= next_stats_ns) {
			emit_stats(skel, "stats");
			next_stats_ns = now + (uint64_t)interval_ms * 1000000ULL;
		}
		nanosleep(&delay, NULL);
	}
	unexpected_exit = !exit_requested && UEI_EXITED(skel, uei);
	emit_stats(skel, "final_stats");
	return_code = unexpected_exit || heartbeat_failed ? 1 : 0;

out:
	bpf_link__destroy(link);
	if (maps_pinned)
		unpin_maps(maps, 4);
	if (link && wait_for_value("/sys/kernel/sched_ext/state", "disabled", 3000)) {
		emit_error("sched_ext did not return to disabled state");
		return_code = 1;
	}
	if (skel) {
		struct user_exit_info *uei = &skel->data->uei;

		fputs("{\"event\":\"exit\",\"requested_signal\":", stdout);
		printf("%d,\"unexpected\":%s,\"exit_code\":%llu,\"reason\":",
		       exit_requested, unexpected_exit ? "true" : "false",
		       (unsigned long long)uei->exit_code);
		json_string(uei->reason);
		fputs(",\"message\":", stdout);
		json_string(uei->msg);
		printf(",\"return_code\":%d}\n", return_code);
		fflush(stdout);
	}
	agentos_scx__destroy(skel);
	return return_code;
}
