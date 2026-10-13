#include "common/perfStats.h"

#include <array>
#include <atomic>
#include <cstdio>

namespace PerfStats {

namespace {

constexpr auto TimerCount   = static_cast<size_t>(Timer::Count);
constexpr auto CounterCount = static_cast<size_t>(Counter::Count);

std::array<std::atomic<uint64_t>, TimerCount>   g_timers {};
std::array<std::atomic<uint64_t>, CounterCount> g_counters {};

uint64_t g_last_frames  = 0;
uint64_t g_total_ms     = 0;
bool     g_header_known = false;
FILE*    g_file         = nullptr;

uint64_t TakeMs(Timer timer) noexcept {
	return g_timers[static_cast<size_t>(timer)].exchange(0, std::memory_order_relaxed) / 1000000u;
}

uint64_t Take(Counter counter) noexcept {
	return g_counters[static_cast<size_t>(counter)].exchange(0, std::memory_order_relaxed);
}

} // namespace

void AddTime(Timer timer, uint64_t nanoseconds) noexcept {
	g_timers[static_cast<size_t>(timer)].fetch_add(nanoseconds, std::memory_order_relaxed);
}

void Add(Counter counter, uint64_t value) noexcept {
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

void Report(uint64_t presented_frames, uint64_t elapsed_ms) noexcept {
	if (elapsed_ms == 0) {
		return;
	}
	if (!g_header_known) {
		g_header_known = true;
		g_file         = std::fopen("_perf.log", "w");
		if (g_file != nullptr) {
			std::fprintf(g_file,
			             "# per-interval totals; times in ms (wall time summed across threads)\n");
		}
		g_last_frames = presented_frames;
	}
	g_total_ms += elapsed_ms;

	const auto frames = presented_frames - g_last_frames;
	g_last_frames     = presented_frames;
	const auto fps    = static_cast<double>(frames) * 1000.0 / static_cast<double>(elapsed_ms);

	const auto readback      = TakeMs(Timer::GuestReadback);
	const auto download      = TakeMs(Timer::DownloadWait);
	const auto finish        = TakeMs(Timer::SchedulerFinish);
	const auto sched_wait    = TakeMs(Timer::SchedulerWait);
	const auto stream        = TakeMs(Timer::StreamWait);
	const auto fault_buf     = TakeMs(Timer::FaultBufferWait);
	const auto flip_slot     = TakeMs(Timer::FlipSlotWait);
	const auto gpu_idle      = TakeMs(Timer::GpuThreadIdle);
	const auto gpu_busy      = TakeMs(Timer::GpuThreadBusy);
	const auto shader        = TakeMs(Timer::ShaderCompile);
	const auto readbacks     = Take(Counter::GuestReadbacks);
	const auto read_faults   = Take(Counter::ReadFaults);
	const auto write_faults  = Take(Counter::WriteFaults);
	const auto shaders       = Take(Counter::ShaderCompiles);
	const auto submissions   = Take(Counter::Submissions);

	if (g_file == nullptr) {
		return;
	}
	std::fprintf(g_file,
	             "t=%6.1fs int=%4llums fps=%5.1f | gpu-thread busy=%4llu idle=%4llu subs=%llu | "
	             "readback n=%llu guest-blocked=%llu gpu-wait=%llu | finish=%llu sched-wait=%llu "
	             "stream=%llu faultbuf=%llu | flip-slot=%llu | shaders n=%llu %llums | "
	             "faults r=%llu w=%llu\n",
	             static_cast<double>(g_total_ms) / 1000.0,
	             static_cast<unsigned long long>(elapsed_ms), fps,
	             static_cast<unsigned long long>(gpu_busy), static_cast<unsigned long long>(gpu_idle),
	             static_cast<unsigned long long>(submissions),
	             static_cast<unsigned long long>(readbacks),
	             static_cast<unsigned long long>(readback),
	             static_cast<unsigned long long>(download),
	             static_cast<unsigned long long>(finish),
	             static_cast<unsigned long long>(sched_wait),
	             static_cast<unsigned long long>(stream),
	             static_cast<unsigned long long>(fault_buf),
	             static_cast<unsigned long long>(flip_slot),
	             static_cast<unsigned long long>(shaders), static_cast<unsigned long long>(shader),
	             static_cast<unsigned long long>(read_faults),
	             static_cast<unsigned long long>(write_faults));
	std::fflush(g_file);
}

} // namespace PerfStats
