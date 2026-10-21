/* SPDX-License-Identifier: GPL-2.0 */
#define _GNU_SOURCE

#include <errno.h>
#include <getopt.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define MAX_LATENCY_SAMPLES 10000U

static uint64_t now_ns(void)
{
	struct timespec now;

	if (clock_gettime(CLOCK_MONOTONIC, &now)) {
		perror("clock_gettime");
		exit(1);
	}
	return (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

static uint64_t mix(uint64_t value)
{
	value ^= value >> 30;
	value *= 0xbf58476d1ce4e5b9ULL;
	value ^= value >> 27;
	value *= 0x94d049bb133111ebULL;
	return value ^ (value >> 31);
}

static void usage(const char *program)
{
	fprintf(stderr,
		"Usage: %s --role latency|normal|batch --duration-ms N "
		"[--cpu CPU | --cpus CPU,CPU,...] "
		"[--period-us N] [--work-us N] [--seed N]\n",
		program);
}

static int parse_cpu_list(const char *value, cpu_set_t *allowed)
{
	char copy[256];
	char *cursor;
	char *token;
	unsigned int count = 0;

	if (strlen(value) >= sizeof(copy))
		return -1;
	strcpy(copy, value);
	CPU_ZERO(allowed);
	cursor = copy;
	while ((token = strsep(&cursor, ",")) != NULL) {
		char *end;
		long cpu;

		if (!*token)
			return -1;
		errno = 0;
		cpu = strtol(token, &end, 10);
		if (errno || *end || cpu < 0 || cpu >= CPU_SETSIZE)
			return -1;
		CPU_SET((int)cpu, allowed);
		count++;
	}
	return count ? 0 : -1;
}

int main(int argc, char **argv)
{
	static const struct option options[] = {
		{"role", required_argument, NULL, 'r'},
		{"duration-ms", required_argument, NULL, 'd'},
		{"cpu", required_argument, NULL, 'c'},
		{"cpus", required_argument, NULL, 'C'},
		{"period-us", required_argument, NULL, 'p'},
		{"work-us", required_argument, NULL, 'w'},
		{"seed", required_argument, NULL, 's'},
		{NULL, 0, NULL, 0},
	};
	const char *role = NULL, *cpu_list = NULL;
	uint64_t duration_ms = 0, period_us = 10000, work_us = 1000, seed = 1;
	uint64_t latencies[MAX_LATENCY_SAMPLES];
	uint64_t checksum, iterations = 0, samples = 0, max_gap_ns = 0;
	uint64_t validation_checksum;
	uint64_t start, deadline, previous_progress;
	struct rusage resource_usage;
	cpu_set_t allowed;
	int cpu = -1, first_cpu = -1, last_cpu = -1, violations = 0;
	int affinity_configured = 0;
	int option;
	char barrier;

	while ((option = getopt_long(argc, argv, "r:d:c:C:p:w:s:", options, NULL)) != -1) {
		switch (option) {
		case 'r': role = optarg; break;
		case 'd': duration_ms = strtoull(optarg, NULL, 10); break;
		case 'c': cpu = atoi(optarg); break;
		case 'C': cpu_list = optarg; break;
		case 'p': period_us = strtoull(optarg, NULL, 10); break;
		case 'w': work_us = strtoull(optarg, NULL, 10); break;
		case 's': seed = strtoull(optarg, NULL, 10); break;
		default: usage(argv[0]); return 2;
		}
	}
	if (!role || !duration_ms || !period_us || !work_us ||
	    (cpu >= 0 && cpu_list) ||
	    (strcmp(role, "latency") && strcmp(role, "normal") && strcmp(role, "batch"))) {
		usage(argv[0]);
		return 2;
	}
	if (cpu >= 0) {
		CPU_ZERO(&allowed);
		CPU_SET(cpu, &allowed);
		affinity_configured = 1;
	} else if (cpu_list) {
		if (parse_cpu_list(cpu_list, &allowed)) {
			fprintf(stderr, "invalid CPU list: %s\n", cpu_list);
			return 2;
		}
		affinity_configured = 1;
	}
	if (affinity_configured &&
	    sched_setaffinity(0, sizeof(allowed), &allowed)) {
		perror("sched_setaffinity");
		return 1;
	}
	if (fputs("READY\n", stderr) == EOF || fflush(stderr)) {
		perror("publish startup readiness");
		return 1;
	}
	if (read(STDIN_FILENO, &barrier, 1) != 1) {
		fprintf(stderr, "startup barrier was not released\n");
		return 1;
	}

	start = now_ns();
	deadline = start + duration_ms * 1000000ULL;
	previous_progress = start;
	checksum = seed;
	validation_checksum = seed;
	for (uint64_t index = 0; index < 1000; index++)
		validation_checksum = mix(validation_checksum + index + 1);
	while (now_ns() < deadline) {
		uint64_t progress_started = now_ns();
		int current_cpu = sched_getcpu();

		if (first_cpu < 0)
			first_cpu = current_cpu;
		last_cpu = current_cpu;
		if (affinity_configured && !CPU_ISSET(current_cpu, &allowed))
			violations++;
		if (!strcmp(role, "latency")) {
			uint64_t target = start + (samples + 1) * period_us * 1000ULL;
			struct timespec sleep_until = {
				.tv_sec = target / 1000000000ULL,
				.tv_nsec = target % 1000000000ULL,
			};
			uint64_t work_deadline;

			if (samples >= MAX_LATENCY_SAMPLES || target >= deadline)
				break;
			clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &sleep_until, NULL);
			work_deadline = now_ns() + work_us * 1000ULL;
			while (now_ns() < work_deadline) {
				checksum = mix(checksum + iterations + 1);
				iterations++;
			}
			latencies[samples++] = now_ns() > target ? now_ns() - target : 0;
		} else {
			uint64_t chunk_end = progress_started + work_us * 1000ULL;

			while (now_ns() < chunk_end) {
				checksum = mix(checksum + iterations + 1);
				iterations++;
			}
		}
		if (progress_started > previous_progress &&
		    max_gap_ns < progress_started - previous_progress)
			max_gap_ns = progress_started - previous_progress;
		previous_progress = progress_started;
	}
	{
		uint64_t finished = now_ns();

		if (finished > previous_progress &&
		    max_gap_ns < finished - previous_progress)
			max_gap_ns = finished - previous_progress;
	}
	if (getrusage(RUSAGE_SELF, &resource_usage)) {
		perror("getrusage");
		return 1;
	}

	printf("{\"role\":\"%s\",\"completed\":true,\"checksum\":%llu,"
	       "\"validation_checksum\":%llu,"
	       "\"started_at_ns\":%llu,"
	       "\"iterations\":%llu,\"samples\":%llu,\"max_gap_us\":%.3f,"
	       "\"first_cpu\":%d,\"last_cpu\":%d,\"required_cpu\":%d,"
	       "\"allowed_cpus\":\"%s\","
	       "\"affinity_violations\":%d,"
	       "\"cpu_user_us\":%llu,\"cpu_system_us\":%llu,"
	       "\"voluntary_context_switches\":%ld,"
	       "\"involuntary_context_switches\":%ld,\"latencies_us\":[",
	       role, (unsigned long long)checksum,
	       (unsigned long long)validation_checksum,
	       (unsigned long long)start,
	       (unsigned long long)iterations,
	       (unsigned long long)samples, max_gap_ns / 1000.0,
	       first_cpu, last_cpu, cpu,
	       cpu_list ? cpu_list : cpu >= 0 ? "single" : "all",
	       violations,
	       (unsigned long long)resource_usage.ru_utime.tv_sec * 1000000ULL +
	       (unsigned long long)resource_usage.ru_utime.tv_usec,
	       (unsigned long long)resource_usage.ru_stime.tv_sec * 1000000ULL +
	       (unsigned long long)resource_usage.ru_stime.tv_usec,
	       resource_usage.ru_nvcsw, resource_usage.ru_nivcsw);
	for (uint64_t index = 0; index < samples; index++)
		printf("%s%.3f", index ? "," : "", latencies[index] / 1000.0);
	puts("]}");
	return 0;
}
