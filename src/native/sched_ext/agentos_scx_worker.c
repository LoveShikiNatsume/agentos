/* SPDX-License-Identifier: GPL-2.0 */
#define _GNU_SOURCE

#include <errno.h>
#include <getopt.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef SCHED_EXT
#define SCHED_EXT 7
#endif

struct worker_result {
	uint64_t checksum;
	uint64_t iterations;
	int first_cpu;
	int last_cpu;
	int affinity_violations;
};

static int read_full(int fd, void *buffer, size_t size)
{
	char *cursor = buffer;
	size_t remaining = size;

	while (remaining) {
		ssize_t count = read(fd, cursor, remaining);

		if (count < 0 && errno == EINTR)
			continue;
		if (count <= 0)
			return -1;
		cursor += count;
		remaining -= (size_t)count;
	}
	return 0;
}

static int write_full(int fd, const void *buffer, size_t size)
{
	const char *cursor = buffer;
	size_t remaining = size;

	while (remaining) {
		ssize_t count = write(fd, cursor, remaining);

		if (count < 0 && errno == EINTR)
			continue;
		if (count <= 0)
			return -1;
		cursor += count;
		remaining -= (size_t)count;
	}
	return 0;
}

static int read_ext_enabled(pid_t pid)
{
	char path[64];
	char line[512];
	FILE *file;
	int enabled = -1;

	snprintf(path, sizeof(path), "/proc/%d/sched", pid);
	file = fopen(path, "re");
	if (!file)
		return -1;
	while (fgets(line, sizeof(line), file)) {
		if (sscanf(line, "ext.enabled : %d", &enabled) == 1)
			break;
	}
	fclose(file);
	return enabled;
}

static uint64_t elapsed_ns(const struct timespec *start,
			   const struct timespec *now)
{
	return (uint64_t)(now->tv_sec - start->tv_sec) * 1000000000ULL +
	       (uint64_t)(now->tv_nsec - start->tv_nsec);
}

static void run_worker(unsigned int duration_ms, int required_cpu,
		       int release_fd, int result_fd)
{
	struct worker_result result = {
		.checksum = 0x9e3779b97f4a7c15ULL,
		.first_cpu = -1,
		.last_cpu = -1,
	};
	struct timespec start, now;
	char release;

	if (read_full(release_fd, &release, 1))
		_exit(3);
	clock_gettime(CLOCK_MONOTONIC, &start);
	do {
		int cpu = sched_getcpu();

		if (result.first_cpu < 0)
			result.first_cpu = cpu;
		result.last_cpu = cpu;
		if (required_cpu >= 0 && cpu != required_cpu)
			result.affinity_violations++;
		result.checksum ^= result.checksum << 7;
		result.checksum ^= result.checksum >> 9;
		result.checksum += 0x9e3779b97f4a7c15ULL + result.iterations;
		result.iterations++;
		clock_gettime(CLOCK_MONOTONIC, &now);
	} while (elapsed_ns(&start, &now) < (uint64_t)duration_ms * 1000000ULL);

	if (!result.checksum)
		result.checksum = 1;
	if (write_full(result_fd, &result, sizeof(result)))
		_exit(4);
	_exit(0);
}

static int parse_uint(const char *value, unsigned int *result)
{
	char *end;
	unsigned long parsed;

	errno = 0;
	parsed = strtoul(value, &end, 10);
	if (errno || *end || parsed == 0 || parsed > 600000)
		return -1;
	*result = (unsigned int)parsed;
	return 0;
}

static int parse_cpu(const char *value, int *result)
{
	char *end;
	long parsed;

	errno = 0;
	parsed = strtol(value, &end, 10);
	if (errno || *end || parsed < 0 || parsed >= CPU_SETSIZE)
		return -1;
	*result = (int)parsed;
	return 0;
}

static void usage(FILE *stream, const char *program)
{
	fprintf(stream, "Usage: %s --duration-ms N [--cpu CPU]\n", program);
}

int main(int argc, char **argv)
{
	static const struct option options[] = {
		{"duration-ms", required_argument, NULL, 'd'},
		{"cpu", required_argument, NULL, 'c'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	unsigned int duration_ms = 0;
	int required_cpu = -1;
	int release_pipe[2];
	int result_pipe[2];
	struct sched_param parameter = {.sched_priority = 0};
	struct worker_result result = {};
	pid_t child;
	pid_t worker_pid;
	int policy_before;
	int policy_after = -1;
	int ext_before;
	int ext_after = -1;
	int parent_ext = -1;
	int status = 0;
	int option;
	bool configured = false;

	while ((option = getopt_long(argc, argv, "d:c:h", options, NULL)) != -1) {
		switch (option) {
		case 'd':
			if (parse_uint(optarg, &duration_ms)) {
				usage(stderr, argv[0]);
				return 2;
			}
			break;
		case 'c':
			if (parse_cpu(optarg, &required_cpu)) {
				usage(stderr, argv[0]);
				return 2;
			}
			break;
		case 'h':
			usage(stdout, argv[0]);
			return 0;
		default:
			usage(stderr, argv[0]);
			return 2;
		}
	}
	if (!duration_ms) {
		usage(stderr, argv[0]);
		return 2;
	}
	if (pipe(release_pipe) || pipe(result_pipe)) {
		perror("pipe");
		return 1;
	}

	child = fork();
	if (child < 0) {
		perror("fork");
		return 1;
	}
	if (!child) {
		close(release_pipe[1]);
		close(result_pipe[0]);
		run_worker(duration_ms, required_cpu, release_pipe[0], result_pipe[1]);
	}
	worker_pid = child;

	close(release_pipe[0]);
	close(result_pipe[1]);
	policy_before = sched_getscheduler(child);
	ext_before = read_ext_enabled(child);
	if (required_cpu >= 0) {
		cpu_set_t cpus;

		CPU_ZERO(&cpus);
		CPU_SET(required_cpu, &cpus);
		if (sched_setaffinity(child, sizeof(cpus), &cpus))
			goto cleanup;
	}
	if (sched_setscheduler(child, SCHED_EXT, &parameter))
		goto cleanup;
	policy_after = sched_getscheduler(child);
	ext_after = read_ext_enabled(child);
	parent_ext = read_ext_enabled(getpid());
	configured = policy_after == SCHED_EXT && ext_after == 1 && parent_ext == 0;
	if (!configured)
		goto cleanup;
	if (write_full(release_pipe[1], "R", 1))
		goto cleanup;
	close(release_pipe[1]);
	release_pipe[1] = -1;
	if (read_full(result_pipe[0], &result, sizeof(result)))
		goto cleanup;
	if (waitpid(child, &status, 0) < 0)
		goto cleanup;
	child = -1;

cleanup:
	if (release_pipe[1] >= 0)
		close(release_pipe[1]);
	close(result_pipe[0]);
	if (child > 0) {
		kill(child, SIGKILL);
		waitpid(child, &status, 0);
	}
	printf("{\"child_pid\":%d,\"policy_before\":%d,\"policy_after\":%d,"
	       "\"ext_enabled_before\":%d,\"ext_enabled_after\":%d,"
	       "\"parent_ext_enabled\":%d,\"configured\":%s,"
	       "\"completed\":%s,\"checksum\":%llu,\"iterations\":%llu,"
	       "\"first_cpu\":%d,\"last_cpu\":%d,"
	       "\"required_cpu\":%d,\"affinity_violations\":%d}\n",
	       worker_pid, policy_before, policy_after, ext_before,
	       ext_after, parent_ext, configured ? "true" : "false",
	       WIFEXITED(status) && WEXITSTATUS(status) == 0 ? "true" : "false",
	       (unsigned long long)result.checksum,
	       (unsigned long long)result.iterations, result.first_cpu,
	       result.last_cpu, required_cpu, result.affinity_violations);
	return configured && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
	       result.checksum && result.iterations && !result.affinity_violations ? 0 : 1;
}
