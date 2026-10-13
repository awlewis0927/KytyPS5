#ifndef COMMON_PERFSTATS_H_
#define COMMON_PERFSTATS_H_

// Lightweight always-on frame-time accounting. Wait sites add the time they block, the window
// thread calls Report() once per second, and each report is appended to _perf.log next to the
// emulator so a run can be diagnosed without a profiler attached.

#include <chrono>
#include <cstdint>

namespace PerfStats {

enum class Timer : uint32_t {
	GuestReadback,   // guest thread blocked in a synchronous GPU->CPU buffer readback
	DownloadWait,    // GPU thread waiting for the GPU before publishing a readback
	SchedulerFinish, // CommandScheduler::Finish (full GPU drain)
	SchedulerWait,   // CommandScheduler::Wait (any targeted tick wait)
	StreamWait,      // stream/staging buffer wrap waiting for the GPU
	FaultBufferWait, // fault-buffer area reuse waiting for the GPU
	FlipSlotWait,    // guest waiting for a free flip-queue slot
	GpuThreadIdle,   // GPU command thread with no submissions to process
	GpuThreadBusy,   // GPU command thread processing guest submissions
	ShaderCompile,   // shader recompile + Vulkan pipeline creation on a cache miss
	AsyncReadbackWait, // guest thread waiting for an asynchronous readback to publish
	Draw,            // GPU thread inside DrawIndex/DrawAuto
	Dispatch,        // GPU thread inside DispatchDirect/DispatchIndirect
	PipelineLookup,  // program + pipeline lookup (includes compiles on a miss)
	BufferSync,      // buffer cache CPU->GPU synchronization (uploads)
	TextureFind,     // texture cache image lookup/creation
	FaultHandling,   // guest-memory page fault handling (all threads)
	Count
};

enum class Counter : uint32_t {
	GuestReadbacks,
	ReadFaults,
	WriteFaults,
	ShaderCompiles,
	Submissions,
	Draws,
	Dispatches,
	ReadbackRetries,
	MetaReadbacks,
	MetaReadbackSkips,
	MetaGpuDetections,
	Count
};

void AddTime(Timer timer, uint64_t nanoseconds) noexcept;
void Add(Counter counter, uint64_t value = 1) noexcept;

// Reads a boolean from _kyty_tweaks.ini next to the emulator ("key=0" / "key=1"); the file is
// parsed once, and missing keys return default_value.
bool TweakEnabled(const char* key, bool default_value) noexcept;

// Records the calling stack (frames inside the emulator image) for a buffer readback, so the
// emulator code that triggers readbacks can be identified. Windows only; no-op elsewhere.
void NoteReadbackStack(bool gpu_thread) noexcept;

// Called by the window thread about once per second with the presented frame total.
void Report(uint64_t presented_frames, uint64_t elapsed_ms) noexcept;

class ScopedTimer {
public:
	explicit ScopedTimer(Timer timer) noexcept: m_timer(timer), m_start(Clock::now()) {}
	~ScopedTimer() {
		AddTime(m_timer, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                           Clock::now() - m_start)
		                                           .count()));
	}
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
	using Clock = std::chrono::steady_clock;
	Timer             m_timer;
	Clock::time_point m_start;
};

} // namespace PerfStats

#endif /* COMMON_PERFSTATS_H_ */
