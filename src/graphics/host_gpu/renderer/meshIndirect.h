#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class Buffer;

// GPU-side indirect draws for guest mesh (NGG) pipelines: converts the guest indexed-indirect
// argument array into VkDrawMeshTasksIndirectCommandEXT records with a small compute dispatch,
// so the draw sizes never have to be read back to the CPU.
class MeshIndirectConverter {
public:
	struct Request {
		vk::Buffer args_buffer;
		uint64_t   args_offset     = 0; // byte offset of draw 0's record (multiple of 4)
		uint32_t   stride          = 0; // bytes between records (multiple of 4)
		uint32_t   max_count       = 0;
		uint32_t   prim_size       = 3;
		uint32_t   prim_step       = 3;
		uint32_t   prims_per_group = 1;
	};
	struct Tasks {
		vk::Buffer buffer;
		uint64_t   offset = 0;
	};

	static constexpr uint32_t TaskStride = 12; // sizeof(VkDrawMeshTasksIndirectCommandEXT)

	MeshIndirectConverter(GraphicContext& graphics, CommandScheduler& scheduler);
	~MeshIndirectConverter();
	KYTY_CLASS_NO_COPY(MeshIndirectConverter);

	// Records the conversion (outside rendering) plus the barrier that makes the task records
	// visible to indirect mesh draws. Returns an empty buffer when the request cannot be served.
	[[nodiscard]] Tasks Record(vk::CommandBuffer command, const Request& request);

private:
	static constexpr uint32_t Capacity = 1u << 16; // task records in the ring

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	vk::DescriptorSetLayout m_desc_layout     = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
	std::unique_ptr<Buffer> m_tasks;
	std::unique_ptr<uint64_t[]> m_ticks; // per 1024-record block: last tick that used it
	uint32_t                m_cursor = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
