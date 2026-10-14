#include "common/perfStats.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

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

std::map<std::string, bool>& Tweaks() {
	static std::map<std::string, bool> tweaks = [] {
		std::map<std::string, bool> result;
		if (FILE* file = std::fopen("_kyty_tweaks.ini", "r"); file != nullptr) {
			char line[256];
			while (std::fgets(line, sizeof(line), file) != nullptr) {
				std::string text(line);
				if (const auto hash = text.find('#'); hash != std::string::npos) {
					text.resize(hash);
				}
				const auto eq = text.find('=');
				if (eq == std::string::npos) {
					continue;
				}
				auto trim = [](std::string value) {
					const auto first = value.find_first_not_of(" \t\r\n");
					const auto last  = value.find_last_not_of(" \t\r\n");
					return first == std::string::npos ? std::string {}
					                                  : value.substr(first, last - first + 1);
				};
				const auto key   = trim(text.substr(0, eq));
				const auto value = trim(text.substr(eq + 1));
				if (!key.empty()) {
					result[key] = !(value == "0" || value == "false" || value == "off");
				}
			}
			std::fclose(file);
		}
		return result;
	}();
	return tweaks;
}

std::mutex                                g_stack_mutex;
std::unordered_map<std::string, uint64_t> g_stacks;
uint64_t                                  g_report_count = 0;

void DumpReadbackStacks() noexcept {
	std::vector<std::pair<std::string, uint64_t>> stacks;
	{
		std::lock_guard lock(g_stack_mutex);
		stacks.assign(g_stacks.begin(), g_stacks.end());
	}
	if (stacks.empty()) {
		return;
	}
	std::sort(stacks.begin(), stacks.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	FILE* file = std::fopen("_readback_stacks.log", "w");
	if (file == nullptr) {
		return;
	}
	std::fprintf(file, "# readback call stacks since start (G = GPU thread, C = guest thread)\n"
	                   "# count | frames as RVAs inside kyty_emulator.exe, innermost first\n");
	for (size_t i = 0; i < stacks.size() && i < 40; i++) {
		std::fprintf(file, "%10llu | %s\n", static_cast<unsigned long long>(stacks[i].second),
		             stacks[i].first.c_str());
	}
	std::fclose(file);
}

} // namespace

void NoteReadbackStack(bool gpu_thread) noexcept {
#if defined(_WIN32)
	static const auto image_base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	static const auto image_end  = [] {
		const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image_base);
		const auto* nt =
		    reinterpret_cast<const IMAGE_NT_HEADERS*>(image_base + static_cast<uintptr_t>(dos->e_lfanew));
		return image_base + nt->OptionalHeader.SizeOfImage;
	}();
	void*      frames[48];
	const auto count = RtlCaptureStackBackTrace(1, 48, frames, nullptr);
	std::string key  = gpu_thread ? "G" : "C";
	int         kept = 0;
	for (USHORT i = 0; i < count && kept < 16; i++) {
		const auto address = reinterpret_cast<uintptr_t>(frames[i]);
		if (address >= image_base && address < image_end) {
			char text[24];
			std::snprintf(text, sizeof(text), " %llx",
			              static_cast<unsigned long long>(address - image_base));
			key += text;
			kept++;
		}
	}
	std::lock_guard lock(g_stack_mutex);
	g_stacks[key]++;
#else
	(void)gpu_thread;
#endif
}

namespace {
struct HotPage {
	uint32_t count   = 0;
	uint32_t threads = 0; // distinct thread ids seen, saturating at a small number
	uint32_t last_thread = 0;
};
std::mutex g_hot_mutex;
std::array<std::unordered_map<uint64_t, HotPage>, static_cast<size_t>(HotKind::Count)> g_hot;

uint32_t CurrentThreadId() noexcept {
#if defined(_WIN32)
	return static_cast<uint32_t>(GetCurrentThreadId());
#else
	return 0;
#endif
}

void WriteHotPages(FILE* file) {
	static constexpr const char* names[] = {"guest-readback", "gpu-readback", "write-fault"};
	std::lock_guard lock(g_hot_mutex);
	for (size_t kind = 0; kind < g_hot.size(); kind++) {
		auto& pages = g_hot[kind];
		if (pages.empty()) {
			continue;
		}
		std::vector<std::pair<uint64_t, HotPage>> sorted(pages.begin(), pages.end());
		const auto top = (std::min<size_t>)(sorted.size(), 5);
		std::partial_sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(top),
		                  sorted.end(), [](const auto& a, const auto& b) {
			                  return a.second.count > b.second.count;
		                  });
		std::fprintf(file, "    hot %s pages=%zu:", names[kind], pages.size());
		for (size_t i = 0; i < top; i++) {
			std::fprintf(file, " %llx*%u(t%u/%u)",
			             static_cast<unsigned long long>(sorted[i].first << 12u),
			             sorted[i].second.count, sorted[i].second.last_thread,
			             sorted[i].second.threads);
		}
		std::fprintf(file, "\n");
		pages.clear();
	}
}
} // namespace

struct WrittenRange {
	uint64_t size  = 0;
	uint32_t count = 0;
};
std::unordered_map<uint64_t, WrittenRange> g_written_ranges;

void WriteWrittenRanges(FILE* file) {
	std::lock_guard lock(g_hot_mutex);
	if (g_written_ranges.empty()) {
		return;
	}
	std::vector<std::pair<uint64_t, WrittenRange>> sorted(g_written_ranges.begin(),
	                                                      g_written_ranges.end());
	const auto top = (std::min<size_t>)(sorted.size(), 6);
	std::partial_sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(top),
	                  sorted.end(), [](const auto& a, const auto& b) {
		                  return a.second.size > b.second.size;
	                  });
	std::fprintf(file, "    hot gpu-written-range ranges=%zu:", g_written_ranges.size());
	for (size_t i = 0; i < top; i++) {
		std::fprintf(file, " %llx+%llx*%u", static_cast<unsigned long long>(sorted[i].first),
		             static_cast<unsigned long long>(sorted[i].second.size),
		             sorted[i].second.count);
	}
	std::fprintf(file, "\n");
	g_written_ranges.clear();
}

struct ImageUpload {
	uint64_t size    = 0;
	uint32_t count   = 0;
	uint32_t reasons = 0;
};
std::unordered_map<uint64_t, ImageUpload> g_image_uploads;

void WriteImageUploads(FILE* file) {
	std::lock_guard lock(g_hot_mutex);
	if (g_image_uploads.empty()) {
		return;
	}
	uint64_t total = 0;
	for (const auto& [address, upload]: g_image_uploads) {
		total += upload.size * upload.count;
	}
	std::vector<std::pair<uint64_t, ImageUpload>> sorted(g_image_uploads.begin(),
	                                                     g_image_uploads.end());
	const auto top = (std::min<size_t>)(sorted.size(), 8);
	std::partial_sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(top),
	                  sorted.end(), [](const auto& a, const auto& b) {
		                  return a.second.size * a.second.count > b.second.size * b.second.count;
	                  });
	std::fprintf(file, "    hot image-upload images=%zu total=%lluKiB:", g_image_uploads.size(),
	             static_cast<unsigned long long>(total >> 10u));
	for (size_t i = 0; i < top; i++) {
		std::fprintf(file, " %llx+%llx*%u/r%u", static_cast<unsigned long long>(sorted[i].first),
		             static_cast<unsigned long long>(sorted[i].second.size),
		             sorted[i].second.count, sorted[i].second.reasons);
	}
	std::fprintf(file, "\n");
	g_image_uploads.clear();
}

void NoteImageUpload(uint64_t address, uint64_t size, uint32_t reason) noexcept {
	std::lock_guard lock(g_hot_mutex);
	auto& upload = g_image_uploads[address];
	upload.size  = (std::max)(upload.size, size);
	upload.count++;
	upload.reasons |= reason;
}

void NoteWrittenRange(uint64_t address, uint64_t size) noexcept {
	std::lock_guard lock(g_hot_mutex);
	auto& range = g_written_ranges[address];
	range.size  = (std::max)(range.size, size);
	range.count++;
}

void NoteHotAddress(HotKind kind, uint64_t address) noexcept {
	const auto thread = CurrentThreadId();
	std::lock_guard lock(g_hot_mutex);
	auto& page = g_hot[static_cast<size_t>(kind)][address >> 12u];
	page.count++;
	if (page.last_thread != thread) {
		page.threads = (std::min)(page.threads + 1u, 99u);
		page.last_thread = thread;
	}
}

bool TweakEnabled(const char* key, bool default_value) noexcept {
	const auto& tweaks = Tweaks();
	const auto  it     = tweaks.find(key);
	return it == tweaks.end() ? default_value : it->second;
}

void AddTime(Timer timer, uint64_t nanoseconds) noexcept {
	g_timers[static_cast<size_t>(timer)].fetch_add(nanoseconds, std::memory_order_relaxed);
}

void Add(Counter counter, uint64_t value) noexcept {
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

namespace {

#if defined(_WIN32)
uintptr_t ImageBase() noexcept {
	static const auto base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	return base;
}
uintptr_t ImageEnd() noexcept {
	static const auto end = [] {
		const auto  base = ImageBase();
		const auto* dos  = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
		const auto* nt   = reinterpret_cast<const IMAGE_NT_HEADERS*>(
            base + static_cast<uintptr_t>(dos->e_lfanew));
		return base + nt->OptionalHeader.SizeOfImage;
	}();
	return end;
}
bool InImage(uint64_t pc) noexcept { return pc >= ImageBase() && pc < ImageEnd(); }

// Walks a captured context with the x64 unwind tables. Frames without unwind data (JIT or guest
// code) end the walk, except a leaf at the top whose return address is read from RSP.
uint32_t UnwindContext(CONTEXT ctx, uint64_t* frames, uint32_t max_frames) noexcept {
	uint32_t n = 0;
	while (n < max_frames) {
		const uint64_t pc = ctx.Rip;
		if (pc == 0) {
			break;
		}
		frames[n++]       = pc;
		DWORD64 base      = 0;
		auto*   function  = RtlLookupFunctionEntry(pc, &base, nullptr);
		if (function == nullptr) {
			if (n == 1 && InImage(pc)) {
				ctx.Rip = *reinterpret_cast<const DWORD64*>(ctx.Rsp);
				ctx.Rsp += 8;
				continue;
			}
			break;
		}
		void*   handler_data = nullptr;
		DWORD64 establisher  = 0;
		RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, pc, function, &ctx, &handler_data, &establisher,
		                 nullptr);
	}
	return n;
}

std::string FramesToText(const uint64_t* frames, uint32_t count, uint32_t max_kept) {
	std::string text;
	uint32_t    kept     = 0;
	bool        external = false;
	for (uint32_t i = 0; i < count && kept < max_kept; i++) {
		char item[24];
		if (InImage(frames[i])) {
			std::snprintf(item, sizeof(item), " %llx",
			              static_cast<unsigned long long>(frames[i] - ImageBase()));
			text += item;
			kept++;
			external = false;
		} else if (!external) {
			text += " x";
			external = true;
		}
	}
	return text;
}

std::mutex                                g_sampler_mutex;
std::unordered_map<std::string, uint64_t> g_sampler_total;
std::unordered_map<std::string, uint64_t> g_sampler_window;
uint64_t                                  g_sampler_samples = 0;
std::atomic_bool                          g_sampler_started {false};

void SamplerLoop(HANDLE thread) noexcept {
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	std::vector<std::pair<std::array<uint64_t, 32>, uint32_t>> batch;
	batch.reserve(256);
	for (;;) {
		Sleep(1);
		if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
			return;
		}
		CONTEXT ctx {};
		ctx.ContextFlags = CONTEXT_FULL;
		std::array<uint64_t, 32> frames {};
		uint32_t                 count = 0;
		if (GetThreadContext(thread, &ctx) != 0) {
			count = UnwindContext(ctx, frames.data(), static_cast<uint32_t>(frames.size()));
		}
		ResumeThread(thread);
		if (count == 0) {
			continue;
		}
		batch.emplace_back(frames, count);
		if (batch.size() >= 64) {
			std::lock_guard lock(g_sampler_mutex);
			for (const auto& [f, c]: batch) {
				auto key = FramesToText(f.data(), c, 20);
				g_sampler_total[key]++;
				g_sampler_window[key]++;
				g_sampler_samples++;
			}
			batch.clear();
		}
	}
}

void WriteTop(FILE* file, const std::unordered_map<std::string, uint64_t>& map, size_t limit) {
	std::vector<std::pair<std::string, uint64_t>> stacks(map.begin(), map.end());
	std::sort(stacks.begin(), stacks.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	for (size_t i = 0; i < stacks.size() && i < limit; i++) {
		std::fprintf(file, "%8llu |%s\n", static_cast<unsigned long long>(stacks[i].second),
		             stacks[i].first.c_str());
	}
}
#endif

void DumpSamplerProfile() noexcept {
#if defined(_WIN32)
	if (!g_sampler_started.load()) {
		return;
	}
	std::unordered_map<std::string, uint64_t> total;
	std::unordered_map<std::string, uint64_t> window;
	uint64_t                                  samples = 0;
	{
		std::lock_guard lock(g_sampler_mutex);
		total   = g_sampler_total;
		window.swap(g_sampler_window);
		samples = g_sampler_samples;
	}
	FILE* file = std::fopen("_gpu_profile.log", "w");
	if (file == nullptr) {
		return;
	}
	std::fprintf(file,
	             "# GPU command thread stack samples (~1 kHz). RVAs inside kyty_emulator.exe, "
	             "innermost first; x = frames outside the exe\n# samples=%llu t=%llus\n"
	             "## last 5 seconds\n",
	             static_cast<unsigned long long>(samples),
	             static_cast<unsigned long long>(g_total_ms / 1000));
	WriteTop(file, window, 40);
	std::fprintf(file, "## since start\n");
	WriteTop(file, total, 2000);
	std::fclose(file);
#endif
}

} // namespace

namespace {
std::atomic_bool g_checkpoints_armed {false};
} // namespace

bool CheckpointsArmed() noexcept {
	return g_checkpoints_armed.load(std::memory_order_relaxed);
}

void StartThreadSampler() noexcept {
#if defined(_WIN32)
	if (!TweakEnabled("gpu_profiler", true) || g_sampler_started.exchange(true)) {
		return;
	}
	HANDLE thread = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread,
	                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
	                    FALSE, 0) == 0) {
		return;
	}
	std::thread(SamplerLoop, thread).detach();
#endif
}

void WriteCrashReport(const char* text, void* native_context) noexcept {
	static std::atomic_uint32_t reports {0};
	if (reports.fetch_add(1) >= 16) {
		return;
	}
	FILE* file = std::fopen("_crash.log", "a");
	if (file == nullptr) {
		return;
	}
	std::fprintf(file, "=== t=%llus thread=%lu\n%s\n", static_cast<unsigned long long>(g_total_ms / 1000),
#if defined(_WIN32)
	             static_cast<unsigned long>(GetCurrentThreadId()),
#else
	             0ul,
#endif
	             text);
#if defined(_WIN32)
	uint64_t frames[48];
	uint32_t count = 0;
	if (native_context != nullptr) {
		count = UnwindContext(*static_cast<CONTEXT*>(native_context), frames, 48);
	} else {
		void* raw[48];
		count = RtlCaptureStackBackTrace(0, 48, raw, nullptr);
		for (uint32_t i = 0; i < count; i++) {
			frames[i] = reinterpret_cast<uint64_t>(raw[i]);
		}
	}
	std::fprintf(file, "stack:%s\n", FramesToText(frames, count, 40).c_str());
#else
	(void)native_context;
#endif
	std::fclose(file);
}

void Report(uint64_t presented_frames, uint64_t elapsed_ms) noexcept {
	if (elapsed_ms == 0) {
		return;
	}
	if (++g_report_count % 10 == 0) {
		DumpReadbackStacks();
	}
	if (g_report_count % 5 == 0) {
		DumpSamplerProfile();
	}
	if (!g_header_known) {
		g_header_known = true;
		g_file         = std::fopen("_perf.log", "w");
		if (g_file != nullptr) {
			std::fprintf(g_file,
			             "# per-interval totals; times in ms (wall time summed across threads)\n"
			             "# tweaks: async_readback=%d meta_readback_rt=%d meta_readback_tex=%d "
			             "gpu_fast_clear=%d async_pipelines=%d relax_edge_pages=%d skip_stable_readbacks=%d\n",
			             TweakEnabled("async_readback", true) ? 1 : 0,
			             TweakEnabled("meta_readback_rt", true) ? 1 : 0,
			             TweakEnabled("meta_readback_tex", true) ? 1 : 0,
			             TweakEnabled("gpu_fast_clear", false) ? 1 : 0,
			             TweakEnabled("async_pipelines", false) ? 1 : 0,
			             TweakEnabled("relax_edge_pages", false) ? 1 : 0,
			             TweakEnabled("skip_stable_readbacks", false) ? 1 : 0);
		}
		g_last_frames = presented_frames;
	}
	g_total_ms += elapsed_ms;

	const auto frames = presented_frames - g_last_frames;
	g_last_frames     = presented_frames;
	const auto fps    = static_cast<double>(frames) * 1000.0 / static_cast<double>(elapsed_ms);
	static uint32_t low_fps_seconds = 0;
	low_fps_seconds = (g_total_ms > 60000 && fps < 4.0) ? low_fps_seconds + 1 : 0;
	if (low_fps_seconds >= 3) {
		g_checkpoints_armed.store(true, std::memory_order_relaxed);
	}

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
	const auto async_wait    = TakeMs(Timer::AsyncReadbackWait);
	const auto draw          = TakeMs(Timer::Draw);
	const auto dispatch      = TakeMs(Timer::Dispatch);
	const auto pipeline      = TakeMs(Timer::PipelineLookup);
	const auto buffer_sync   = TakeMs(Timer::BufferSync);
	const auto texture_find  = TakeMs(Timer::TextureFind);
	const auto fault_time    = TakeMs(Timer::FaultHandling);
	const auto draws         = Take(Counter::Draws);
	const auto dispatches    = Take(Counter::Dispatches);
	const auto retries       = Take(Counter::ReadbackRetries);
	const auto meta_reads    = Take(Counter::MetaReadbacks);
	const auto meta_skips    = Take(Counter::MetaReadbackSkips);
	const auto meta_gpu      = Take(Counter::MetaGpuDetections);
	const auto meta_cached   = Take(Counter::MetaGpuCached);
	const auto srt           = TakeMs(Timer::SrtMaterialize);
	const auto pipe_state    = TakeMs(Timer::PipelineState);
	const auto bindings      = TakeMs(Timer::Bindings);
	const auto targets       = TakeMs(Timer::RenderTargets);
	const auto commit        = TakeMs(Timer::Commit);
	const auto unmap_drains  = Take(Counter::UnmapDrains);
	const auto indirect      = Take(Counter::IndirectDraws);
	const auto async_pipes   = Take(Counter::AsyncPipelines);
	const auto async_skips   = Take(Counter::AsyncSkippedDraws);
	const auto relaxed       = Take(Counter::RelaxedFaults);
	const auto stable        = Take(Counter::StableSkips);
	const auto gpu_indirect  = Take(Counter::GpuIndirectDraws);

	if (g_file == nullptr) {
		return;
	}
	std::fprintf(g_file,
	             "t=%6.1fs int=%4llums fps=%5.1f | gpu-thread busy=%4llu idle=%4llu subs=%llu | "
	             "readback n=%llu guest-blocked=%llu gpu-wait=%llu | finish=%llu sched-wait=%llu "
	             "stream=%llu faultbuf=%llu | flip-slot=%llu | shaders n=%llu %llums | "
	             "faults r=%llu w=%llu %llums | async-wait=%llu retries=%llu | draw n=%llu %llums "
	             "dispatch n=%llu %llums | pipeline=%llu bufsync=%llu texfind=%llu | "
	             "meta read=%llu skip=%llu gpu=%llu cached=%llu | srt=%llu pipe=%llu bind=%llu rt=%llu "
	             "commit=%llu | unmap-drain=%llu indirect=%llu | async-pipes=%llu skipped=%llu | relaxed=%llu stable=%llu gpu-indirect=%llu\n",
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
	             static_cast<unsigned long long>(write_faults),
	             static_cast<unsigned long long>(fault_time),
	             static_cast<unsigned long long>(async_wait), static_cast<unsigned long long>(retries),
	             static_cast<unsigned long long>(draws), static_cast<unsigned long long>(draw),
	             static_cast<unsigned long long>(dispatches),
	             static_cast<unsigned long long>(dispatch),
	             static_cast<unsigned long long>(pipeline),
	             static_cast<unsigned long long>(buffer_sync),
	             static_cast<unsigned long long>(texture_find),
	             static_cast<unsigned long long>(meta_reads),
	             static_cast<unsigned long long>(meta_skips),
	             static_cast<unsigned long long>(meta_gpu),
	             static_cast<unsigned long long>(meta_cached), static_cast<unsigned long long>(srt),
	             static_cast<unsigned long long>(pipe_state),
	             static_cast<unsigned long long>(bindings), static_cast<unsigned long long>(targets),
	             static_cast<unsigned long long>(commit),
	             static_cast<unsigned long long>(unmap_drains),
	             static_cast<unsigned long long>(indirect),
	             static_cast<unsigned long long>(async_pipes),
	             static_cast<unsigned long long>(async_skips),
	             static_cast<unsigned long long>(relaxed), static_cast<unsigned long long>(stable),
	             static_cast<unsigned long long>(gpu_indirect));
	WriteHotPages(g_file);
	WriteWrittenRanges(g_file);
	WriteImageUploads(g_file);
	std::fflush(g_file);
}

} // namespace PerfStats
