#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_METACLEARDETECTOR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_METACLEARDETECTOR_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class Buffer;

// GPU-side detection of guest color fast clears (DCC/CMask metadata).
//
// The CPU path reads GPU-written metadata back to decide whether a render target or texture
// slice was fast-cleared, which drains the whole GPU queue on the emulator's GPU thread. This
// helper records a small compute dispatch instead: it scans each metadata slice on the GPU and
// writes one predicate word per candidate clear code, which the caller consumes with
// VK_EXT_conditional_rendering so the matching clear runs without any CPU synchronization.
class MetaClearDetector {
public:
	static constexpr uint32_t Candidates = 5;
	static constexpr uint32_t SlotDwords = 8;   // predicate dwords reserved per slice
	static constexpr uint32_t Capacity   = 4096; // predicate slots in the ring
	static constexpr uint32_t MaxSlices  = 256;

	struct Request {
		vk::Buffer                       meta_buffer;
		uint64_t                         meta_offset  = 0; // byte offset of the first slice
		uint64_t                         slice_size   = 0; // bytes per slice (multiple of 4)
		uint32_t                         slice_count  = 0;
		std::array<uint8_t, Candidates>  codes        = {};
		uint32_t                         enabled_mask = 0;
		bool                             consume      = false;
	};

	struct Predicates {
		vk::Buffer buffer;
		uint64_t   offset = 0; // byte offset of slice 0, candidate 0
		[[nodiscard]] uint64_t Offset(uint32_t slice, uint32_t candidate) const {
			return offset + (static_cast<uint64_t>(slice) * SlotDwords + candidate) * 4u;
		}
	};

	MetaClearDetector(GraphicContext& graphics, CommandScheduler& scheduler);
	~MetaClearDetector();
	KYTY_CLASS_NO_COPY(MetaClearDetector);

	// Records the detection dispatch plus the barriers that make its results visible to
	// conditional rendering and to later metadata consumers.
	[[nodiscard]] Predicates Record(vk::CommandBuffer command, const Request& request);

private:
	[[nodiscard]] uint32_t Allocate(uint32_t count);

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	vk::DescriptorSetLayout m_desc_layout     = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
	std::unique_ptr<Buffer> m_predicates;
	std::vector<uint64_t>   m_slot_ticks;
	uint32_t                m_cursor = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_METACLEARDETECTOR_H_
