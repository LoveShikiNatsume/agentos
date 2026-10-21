/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Minimal AgentOS partial-mode scheduler derived from Linux scx_simple at
 * linux-6.12.y commit 296aabce459470a4c1b68ffd0c0c0920e563aaad.
 */
#include <scx/common.bpf.h>

#include "agentos_scx.h"

char _license[] SEC("license") = "GPL";

UEI_DEFINE(uei);

static u64 normal_vtime_now;
static u64 batch_vtime_now;
static u32 latency_burst;

enum policy_fallback_reason {
	POLICY_OK,
	POLICY_LOOKUP,
	POLICY_VERSION,
	POLICY_RANGE,
	POLICY_GENERATION,
	POLICY_HEARTBEAT,
	POLICY_STALE,
};

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u64));
	__uint(max_entries, AGENTOS_SCX_STAT_NR);
} stats SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(key_size, sizeof(struct agentos_task_key));
	__uint(value_size, sizeof(struct agentos_task_policy_v1));
	__uint(max_entries, AGENTOS_SCX_MAX_TASKS);
} task_policy_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(key_size, sizeof(struct agentos_task_key));
	__uint(value_size, sizeof(struct agentos_task_state_v1));
	__uint(max_entries, AGENTOS_SCX_MAX_TASKS);
} task_state_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(struct agentos_global_config_v1));
	__uint(max_entries, 1);
} global_config_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u64));
	__uint(max_entries, AGENTOS_SCX_MAX_CPUS);
} preempt_state SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u64));
	__uint(max_entries, 1);
} batch_dispatch_state SEC(".maps");

static void stat_inc(u32 index)
{
	u64 *value = bpf_map_lookup_elem(&stats, &index);

	if (value)
		(*value)++;
}

static void stat_add(u32 index, u64 amount)
{
	u64 *value = bpf_map_lookup_elem(&stats, &index);

	if (value)
		*value += amount;
}

static void stat_max(u32 index, u64 candidate)
{
	u64 *value = bpf_map_lookup_elem(&stats, &index);

	if (value && *value < candidate)
		*value = candidate;
}

static bool vtime_before(u64 a, u64 b)
{
	return (s64)(a - b) < 0;
}

static u64 slice_ns(const struct agentos_task_policy_v1 *policy)
{
	u64 slice_us = policy->slice_us;

	if (slice_us < AGENTOS_SCX_SLICE_US_MIN)
		slice_us = AGENTOS_SCX_SLICE_US_MIN;
	if (slice_us > AGENTOS_SCX_SLICE_US_MAX)
		slice_us = AGENTOS_SCX_SLICE_US_MAX;
	return slice_us * 1000ULL;
}

static void default_policy(struct agentos_task_policy_v1 *policy)
{
	__builtin_memset(policy, 0, sizeof(*policy));
	policy->version = AGENTOS_SCX_POLICY_SCHEMA_VERSION;
	policy->queue_class = AGENTOS_SCX_QUEUE_NORMAL;
	policy->weight = AGENTOS_SCX_DEFAULT_WEIGHT;
	policy->slice_us = AGENTOS_SCX_NORMAL_SLICE_US;
	policy->pressure_level = AGENTOS_SCX_PRESSURE_NORMAL;
	policy->flags = AGENTOS_SCX_POLICY_FLAG_ADMITTED |
			AGENTOS_SCX_POLICY_FLAG_FALLBACK;
}

static enum policy_fallback_reason resolve_policy(
	struct task_struct *p, struct agentos_task_policy_v1 *resolved, u64 now)
{
	u32 config_key = 0;
	struct agentos_global_config_v1 *config;
	struct agentos_task_key key = {
		.tgid = p->tgid,
		.pid = p->pid,
	};
	struct agentos_task_policy_v1 *policy;
	u64 heartbeat_timeout_ns;

	default_policy(resolved);
	policy = bpf_map_lookup_elem(&task_policy_map, &key);
	if (!policy)
		return POLICY_LOOKUP;
	if (policy->version != AGENTOS_SCX_POLICY_SCHEMA_VERSION)
		return POLICY_VERSION;
	if (policy->queue_class < AGENTOS_SCX_QUEUE_LATENCY ||
	    policy->queue_class > AGENTOS_SCX_QUEUE_BATCH ||
	    policy->weight < AGENTOS_SCX_WEIGHT_MIN ||
	    policy->weight > AGENTOS_SCX_WEIGHT_MAX ||
	    policy->slice_us < AGENTOS_SCX_SLICE_US_MIN ||
	    policy->slice_us > AGENTOS_SCX_SLICE_US_MAX ||
	    policy->pressure_level < AGENTOS_SCX_PRESSURE_NORMAL ||
	    policy->pressure_level > AGENTOS_SCX_PRESSURE_OVERLOADED ||
	    (policy->flags & ~AGENTOS_SCX_POLICY_KNOWN_FLAGS))
		return POLICY_RANGE;
	config = bpf_map_lookup_elem(&global_config_map, &config_key);
	if (!config ||
	    config->schema_version != AGENTOS_SCX_POLICY_SCHEMA_VERSION ||
	    policy->generation != config->generation)
		return POLICY_GENERATION;
	heartbeat_timeout_ns = (u64)(config->reserved ? config->reserved :
					   AGENTOS_SCX_HEARTBEAT_TIMEOUT_SEC) *
			       1000000000ULL;
	if (now > config->heartbeat_ns &&
	    now - config->heartbeat_ns > heartbeat_timeout_ns)
		return POLICY_HEARTBEAT;
	if (!policy->updated_at_ns ||
	    (now > policy->updated_at_ns &&
	     now - policy->updated_at_ns > heartbeat_timeout_ns))
		return POLICY_STALE;
	__builtin_memcpy(resolved, policy, sizeof(*resolved));
	return POLICY_OK;
}

static void record_policy_result(enum policy_fallback_reason reason)
{
	if (reason == POLICY_OK) {
		stat_inc(AGENTOS_SCX_STAT_POLICY_HIT);
		return;
	}
	stat_inc(AGENTOS_SCX_STAT_UNKNOWN_POLICY);
	stat_inc(AGENTOS_SCX_STAT_FALLBACK);
	if (reason == POLICY_LOOKUP)
		stat_inc(AGENTOS_SCX_STAT_FALLBACK_LOOKUP);
	else if (reason == POLICY_VERSION)
		stat_inc(AGENTOS_SCX_STAT_FALLBACK_VERSION);
	else if (reason == POLICY_RANGE)
		stat_inc(AGENTOS_SCX_STAT_FALLBACK_RANGE);
	else if (reason == POLICY_GENERATION)
		stat_inc(AGENTOS_SCX_STAT_FALLBACK_GENERATION);
	else if (reason == POLICY_HEARTBEAT)
		stat_inc(AGENTOS_SCX_STAT_FALLBACK_HEARTBEAT);
	else if (reason == POLICY_STALE)
		stat_inc(AGENTOS_SCX_STAT_FALLBACK_POLICY_STALE);
}

static struct agentos_task_state_v1 *task_state(struct task_struct *p, bool create)
{
	struct agentos_task_key key = {
		.tgid = p->tgid,
		.pid = p->pid,
	};
	struct agentos_task_state_v1 initial = {
		.last_cpu = (__u32)-1,
	};
	struct agentos_task_state_v1 *state;

	state = bpf_map_lookup_elem(&task_state_map, &key);
	if (!state && create) {
		bpf_map_update_elem(&task_state_map, &key, &initial, BPF_NOEXIST);
		state = bpf_map_lookup_elem(&task_state_map, &key);
	}
	return state;
}

static void record_enqueue(u32 queue)
{
	if (queue == AGENTOS_SCX_QUEUE_LATENCY)
		stat_inc(AGENTOS_SCX_STAT_ENQUEUE_LATENCY);
	else if (queue == AGENTOS_SCX_QUEUE_BATCH)
		stat_inc(AGENTOS_SCX_STAT_ENQUEUE_BATCH);
	else
		stat_inc(AGENTOS_SCX_STAT_ENQUEUE_NORMAL);
}

static void record_direct(u32 queue)
{
	if (queue == AGENTOS_SCX_QUEUE_LATENCY)
		stat_inc(AGENTOS_SCX_STAT_DIRECT_LATENCY);
	else if (queue == AGENTOS_SCX_QUEUE_BATCH)
		stat_inc(AGENTOS_SCX_STAT_DIRECT_BATCH);
	else
		stat_inc(AGENTOS_SCX_STAT_DIRECT_NORMAL);
}

static void request_latency_preempt(struct task_struct *p, u64 now)
{
	s32 cpu = scx_bpf_task_cpu(p);
	u32 key;
	u64 *last;

	if (cpu < 0 || cpu >= AGENTOS_SCX_MAX_CPUS) {
		stat_inc(AGENTOS_SCX_STAT_PREEMPT_SUPPRESSED);
		return;
	}
	key = (u32)cpu;
	last = bpf_map_lookup_elem(&preempt_state, &key);

	if (!last || (now > *last && now - *last < AGENTOS_SCX_PREEMPT_INTERVAL_NS)) {
		stat_inc(AGENTOS_SCX_STAT_PREEMPT_SUPPRESSED);
		return;
	}
	*last = now;
	stat_inc(AGENTOS_SCX_STAT_PREEMPT_REQUESTED);
	scx_bpf_kick_cpu(cpu, SCX_KICK_PREEMPT);
}

static void kick_task_cpu_if_idle(struct task_struct *p)
{
	s32 cpu = scx_bpf_task_cpu(p);

	if (cpu >= 0)
		scx_bpf_kick_cpu(cpu, SCX_KICK_IDLE);
}

s32 BPF_STRUCT_OPS(agentos_scx_select_cpu, struct task_struct *p,
			   s32 prev_cpu, u64 wake_flags)
{
	bool is_idle = false;
	s32 cpu;
	u64 now = bpf_ktime_get_ns();
	struct agentos_task_policy_v1 policy;
	enum policy_fallback_reason reason;
	struct agentos_task_state_v1 *state;

	stat_inc(AGENTOS_SCX_STAT_SELECT_CPU);
	cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
	if (is_idle) {
		reason = resolve_policy(p, &policy, now);
		record_policy_result(reason);
		/*
		 * Latency tasks must be observed by the shared latency DSQ even when
		 * select_cpu finds an idle target.  Direct-local dispatch here made
		 * latency DSQ consumption depend on incidental CPU occupancy and left
		 * full-mode trials with direct_latency > 0 but dispatch_latency == 0.
		 * Returning without dispatch lets enqueue() place the task on the
		 * latency DSQ and kick the selected CPU.
		 */
		if (policy.queue_class == AGENTOS_SCX_QUEUE_LATENCY)
			return cpu;
		state = task_state(p, true);
		if (state) {
			state->runnable_at_ns = now;
			state->flags = policy.queue_class;
		} else {
			stat_inc(AGENTOS_SCX_STAT_FALLBACK_STATE);
		}
		stat_inc(AGENTOS_SCX_STAT_DIRECT_LOCAL);
		record_direct(policy.queue_class);
		scx_bpf_dispatch(p, SCX_DSQ_LOCAL, slice_ns(&policy), 0);
	}

	return cpu;
}

void BPF_STRUCT_OPS(agentos_scx_enqueue, struct task_struct *p,
			    u64 enq_flags)
{
	u64 now = bpf_ktime_get_ns();
	struct agentos_task_policy_v1 policy;
	enum policy_fallback_reason reason;
	struct agentos_task_state_v1 *state;
	u64 *queue_vtime;
	u64 vtime;
	u64 slice;
	bool was_runnable;

	stat_inc(AGENTOS_SCX_STAT_ENQUEUE);
	reason = resolve_policy(p, &policy, now);
	record_policy_result(reason);
	record_enqueue(policy.queue_class);
	slice = slice_ns(&policy);
	state = task_state(p, true);
	was_runnable = state &&
		(state->flags & AGENTOS_SCX_TASK_FLAG_CONTINUOUSLY_RUNNABLE);
	if (state) {
		state->runnable_at_ns = now;
		if ((state->flags & 0xffU) != policy.queue_class)
			state->vtime = 0;
		state->flags = policy.queue_class;
	} else {
		stat_inc(AGENTOS_SCX_STAT_FALLBACK_STATE);
	}

	if (policy.queue_class == AGENTOS_SCX_QUEUE_LATENCY) {
		scx_bpf_dispatch(p, AGENTOS_SCX_LATENCY_DSQ, slice, enq_flags);
		kick_task_cpu_if_idle(p);
		if (!was_runnable)
			request_latency_preempt(p, now);
		return;
	}
	queue_vtime = policy.queue_class == AGENTOS_SCX_QUEUE_BATCH ?
			 &batch_vtime_now : &normal_vtime_now;
	vtime = state && state->vtime ? state->vtime : *queue_vtime;
	if (*queue_vtime > slice &&
	    vtime_before(vtime, *queue_vtime - slice))
		vtime = *queue_vtime - slice;
	if (state)
		state->vtime = vtime;
	scx_bpf_dispatch_vtime(
		p,
		policy.queue_class == AGENTOS_SCX_QUEUE_BATCH ?
			AGENTOS_SCX_BATCH_DSQ : AGENTOS_SCX_NORMAL_DSQ,
		slice, vtime, enq_flags
	);
	kick_task_cpu_if_idle(p);
}

void BPF_STRUCT_OPS(agentos_scx_dispatch, s32 cpu, struct task_struct *prev)
{
	u64 now = bpf_ktime_get_ns();
	u32 state_key = 0;
	u64 *last_batch_dispatch_ns =
		bpf_map_lookup_elem(&batch_dispatch_state, &state_key);
	(void)prev;
	stat_inc(AGENTOS_SCX_STAT_DISPATCH);
	if (last_batch_dispatch_ns &&
	    (!*last_batch_dispatch_ns ||
	     now - *last_batch_dispatch_ns >= AGENTOS_SCX_BATCH_WAIT_NS) &&
	    scx_bpf_consume(AGENTOS_SCX_BATCH_DSQ)) {
		*last_batch_dispatch_ns = now;
		latency_burst = 0;
		stat_inc(AGENTOS_SCX_STAT_DISPATCH_BATCH);
		stat_inc(AGENTOS_SCX_STAT_BATCH_BOOST);
		return;
	}
	if (latency_burst >= AGENTOS_SCX_MAX_LATENCY_BURST) {
		if (scx_bpf_consume(AGENTOS_SCX_NORMAL_DSQ)) {
			latency_burst = 0;
			stat_inc(AGENTOS_SCX_STAT_DISPATCH_NORMAL);
			return;
		}
		if (scx_bpf_consume(AGENTOS_SCX_BATCH_DSQ)) {
			if (last_batch_dispatch_ns)
				*last_batch_dispatch_ns = now;
			latency_burst = 0;
			stat_inc(AGENTOS_SCX_STAT_DISPATCH_BATCH);
			return;
		}
		latency_burst = 0;
	}
	if (scx_bpf_consume(AGENTOS_SCX_LATENCY_DSQ)) {
		latency_burst++;
		stat_inc(AGENTOS_SCX_STAT_DISPATCH_LATENCY);
		return;
	}
	if (scx_bpf_consume(AGENTOS_SCX_NORMAL_DSQ)) {
		latency_burst = 0;
		stat_inc(AGENTOS_SCX_STAT_DISPATCH_NORMAL);
		return;
	}
	if (scx_bpf_consume(AGENTOS_SCX_BATCH_DSQ)) {
		if (last_batch_dispatch_ns)
			*last_batch_dispatch_ns = now;
		latency_burst = 0;
		stat_inc(AGENTOS_SCX_STAT_DISPATCH_BATCH);
	}
	(void)cpu;
}

void BPF_STRUCT_OPS(agentos_scx_running, struct task_struct *p)
{
	u64 now = bpf_ktime_get_ns();
	struct agentos_task_state_v1 *state = task_state(p, true);
	u64 wait_ns = 0;
	u32 queue;
	u32 cpu = bpf_get_smp_processor_id();
	u32 total_index;
	u32 max_index;

	stat_inc(AGENTOS_SCX_STAT_RUNNING);
	if (!state) {
		stat_inc(AGENTOS_SCX_STAT_FALLBACK_STATE);
		return;
	}
	queue = state->flags & 0xffU;
	if (state->runnable_at_ns && now > state->runnable_at_ns)
		wait_ns = now - state->runnable_at_ns;
	state->wait_total_ns += wait_ns;
	if (state->wait_max_ns < wait_ns)
		state->wait_max_ns = wait_ns;
	state->runnable_at_ns = 0;
	state->running_at_ns = now;
	state->dispatches++;
	if (state->last_cpu != (__u32)-1 && state->last_cpu != cpu)
		stat_inc(AGENTOS_SCX_STAT_MIGRATION);
	state->last_cpu = cpu;
	if (queue == AGENTOS_SCX_QUEUE_LATENCY) {
		total_index = AGENTOS_SCX_STAT_WAIT_TOTAL_LATENCY_NS;
		max_index = AGENTOS_SCX_STAT_WAIT_MAX_LATENCY_NS;
	} else if (queue == AGENTOS_SCX_QUEUE_BATCH) {
		total_index = AGENTOS_SCX_STAT_WAIT_TOTAL_BATCH_NS;
		max_index = AGENTOS_SCX_STAT_WAIT_MAX_BATCH_NS;
		if (vtime_before(batch_vtime_now, p->scx.dsq_vtime))
			batch_vtime_now = p->scx.dsq_vtime;
	} else {
		total_index = AGENTOS_SCX_STAT_WAIT_TOTAL_NORMAL_NS;
		max_index = AGENTOS_SCX_STAT_WAIT_MAX_NORMAL_NS;
		if (vtime_before(normal_vtime_now, p->scx.dsq_vtime))
			normal_vtime_now = p->scx.dsq_vtime;
	}
	stat_add(total_index, wait_ns);
	stat_max(max_index, wait_ns);
}

void BPF_STRUCT_OPS(agentos_scx_stopping, struct task_struct *p, bool runnable)
{
	u64 now = bpf_ktime_get_ns();
	struct agentos_task_state_v1 *state = task_state(p, false);
	struct agentos_task_policy_v1 policy;
	u64 runtime_ns;
	u32 runtime_index;

	stat_inc(AGENTOS_SCX_STAT_STOPPING);
	if (!state || !state->running_at_ns || now <= state->running_at_ns)
		return;
	resolve_policy(p, &policy, now);
	runtime_ns = now - state->running_at_ns;
	state->runtime_total_ns += runtime_ns;
	state->running_at_ns = 0;
	state->vtime += runtime_ns * AGENTOS_SCX_DEFAULT_WEIGHT / policy.weight;
	p->scx.dsq_vtime = state->vtime;
	if (policy.queue_class == AGENTOS_SCX_QUEUE_LATENCY)
		runtime_index = AGENTOS_SCX_STAT_RUNTIME_LATENCY_NS;
	else if (policy.queue_class == AGENTOS_SCX_QUEUE_BATCH)
		runtime_index = AGENTOS_SCX_STAT_RUNTIME_BATCH_NS;
	else
		runtime_index = AGENTOS_SCX_STAT_RUNTIME_NORMAL_NS;
	stat_add(runtime_index, runtime_ns);
	if (runnable) {
		state->runnable_at_ns = now;
		state->flags |= AGENTOS_SCX_TASK_FLAG_CONTINUOUSLY_RUNNABLE;
	} else {
		state->flags &= ~AGENTOS_SCX_TASK_FLAG_CONTINUOUSLY_RUNNABLE;
	}
}

void BPF_STRUCT_OPS(agentos_scx_enable, struct task_struct *p)
{
	p->scx.dsq_vtime = normal_vtime_now;
}

void BPF_STRUCT_OPS(agentos_scx_exit_task, struct task_struct *p,
			    struct scx_exit_task_args *args)
{
	struct agentos_task_key key = {
		.tgid = p->tgid,
		.pid = p->pid,
	};

	(void)args;
	bpf_map_delete_elem(&task_state_map, &key);
}

s32 BPF_STRUCT_OPS_SLEEPABLE(agentos_scx_init)
{
	s32 ret;

	ret = scx_bpf_create_dsq(AGENTOS_SCX_LATENCY_DSQ, -1);
	if (ret)
		return ret;
	ret = scx_bpf_create_dsq(AGENTOS_SCX_NORMAL_DSQ, -1);
	if (ret)
		return ret;
	return scx_bpf_create_dsq(AGENTOS_SCX_BATCH_DSQ, -1);
}

void BPF_STRUCT_OPS(agentos_scx_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

SCX_OPS_DEFINE(agentos_scx_ops,
	       .select_cpu	= (void *)agentos_scx_select_cpu,
	       .enqueue		= (void *)agentos_scx_enqueue,
	       .dispatch	= (void *)agentos_scx_dispatch,
	       .running		= (void *)agentos_scx_running,
	       .stopping	= (void *)agentos_scx_stopping,
	       .enable		= (void *)agentos_scx_enable,
	       .exit_task	= (void *)agentos_scx_exit_task,
	       .init		= (void *)agentos_scx_init,
	       .exit		= (void *)agentos_scx_exit,
	       .flags		= SCX_OPS_SWITCH_PARTIAL,
	       .timeout_ms	= AGENTOS_SCX_TIMEOUT_MS,
	       .name		= AGENTOS_SCX_NAME);
